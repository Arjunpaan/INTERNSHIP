#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <thread>
#include <atomic>
#include <iostream>
#include <chrono>

// Pipeline data structure
struct PipelineData {
    GstAppSrc *appsrc;
    cv::Mat latest_frame;
    std::timed_mutex frame_mutex;
};

// Shared data for threading
std::mutex canny_mutex;
cv::Mat latest_canny;
std::atomic<bool> canny_ready(false);
std::atomic<bool> running(true);

// Homography validation (unchanged)
bool validate_homography(const cv::Mat& H, const std::vector<cv::Point2f>& src_pts, const std::vector<cv::Point2f>& dst_pts) {
    if (H.empty()) return false;
    double det = cv::determinant(H);
    if (det < 0.05 || det > 20) return false;
    double rotation = atan2(H.at<double>(1,0), H.at<double>(0,0));
    if (fabs(rotation) > CV_PI/2) return false;
    std::vector<cv::Point2f> src_proj;
    cv::perspectiveTransform(src_pts, src_proj, H);
    double error = 0;
    for (size_t i = 0; i < src_proj.size(); ++i)
        error += cv::norm(src_proj[i] - dst_pts[i]);
    error /= src_proj.size();
    double error_threshold = 8.0;
    if (error > error_threshold) return false;
    return true;
}

// Threaded homography & motion detection (unchanged logic)
void run_homography_and_motion_detection(const cv::Mat& current_frame, cv::Mat& canny_out) {
    static cv::Mat local_prev_frame, local_prev_gray, local_prev_kp_descriptor, local_last_valid_H = cv::Mat::eye(3, 3, CV_64F);
    static std::vector<cv::KeyPoint> local_prev_kp;

    if (local_prev_frame.empty()) {
        local_prev_frame = current_frame.clone();
        cv::cvtColor(local_prev_frame, local_prev_gray, cv::COLOR_BGR2GRAY);
        cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
        orb->detectAndCompute(local_prev_gray, cv::noArray(), local_prev_kp, local_prev_kp_descriptor);
        canny_out = cv::Mat::zeros(current_frame.size(), CV_8UC1);
        return;
    }

    cv::Mat current_gray;
    cv::cvtColor(current_frame, current_gray, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> curr_kp;
    cv::Mat curr_descriptor;
    orb->detectAndCompute(current_gray, cv::noArray(), curr_kp, curr_descriptor);

    std::vector<cv::DMatch> good;
    if (!local_prev_kp.empty() && !curr_kp.empty() && !local_prev_kp_descriptor.empty() && !curr_descriptor.empty()) {
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> matches;
        matcher.knnMatch(local_prev_kp_descriptor, curr_descriptor, matches, 2);
        for (const auto& m : matches)
            if (m[0].distance < 0.75 * m[1].distance)
                good.push_back(m[0]);
    }

    cv::Mat H = local_last_valid_H;
    if (good.size() >= 8) {
        std::vector<cv::Point2f> src_pts, dst_pts;
        for (const auto& m : good) {
            src_pts.push_back(local_prev_kp[m.queryIdx].pt);
            dst_pts.push_back(curr_kp[m.trainIdx].pt);
        }
        cv::Mat H_est = cv::findHomography(src_pts, dst_pts, cv::RANSAC, 4.0);
        if (validate_homography(H_est, src_pts, dst_pts)) {
            H = H_est;
            local_last_valid_H = H;
        }
    }

    cv::Mat aligned_prev;
    cv::warpPerspective(local_prev_frame, aligned_prev, H, current_frame.size());
    cv::Mat prev_mask = cv::Mat::ones(local_prev_frame.size(), CV_8UC1) * 255;
    cv::Mat warped_prev_mask;
    cv::warpPerspective(prev_mask, warped_prev_mask, H, current_frame.size());
    cv::Mat curr_mask = cv::Mat::ones(current_frame.size(), CV_8UC1) * 255;
    cv::Mat valid_mask;
    cv::bitwise_and(warped_prev_mask, curr_mask, valid_mask);

    cv::Mat gray_curr, gray_prev, diff;
    cv::cvtColor(current_frame, gray_curr, cv::COLOR_BGR2GRAY);
    cv::cvtColor(aligned_prev, gray_prev, cv::COLOR_BGR2GRAY);
    cv::subtract(gray_curr, gray_prev, diff);
    cv::bitwise_and(diff, diff, diff, valid_mask);
    cv::GaussianBlur(diff, diff, cv::Size(5,5), 0);
    cv::Mat canny;
    cv::Canny(diff, canny, 50, 150);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::dilate(canny, canny, kernel);

    canny_out = canny.clone();

    local_prev_frame = current_frame.clone();
    local_prev_kp = curr_kp;
    local_prev_kp_descriptor = curr_descriptor;
}

// Thread function
void homography_thread_func(PipelineData* data) {
    while (running) {
        cv::Mat frame_to_process;
        {
            std::unique_lock<std::timed_mutex> lock(data->frame_mutex, std::chrono::milliseconds(3));
            if (!lock.owns_lock() || data->latest_frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            frame_to_process = data->latest_frame.clone();
        }
        cv::Mat canny;
        run_homography_and_motion_detection(frame_to_process, canny);
        {
            std::lock_guard<std::mutex> lock(canny_mutex);
            latest_canny = canny.clone();
            canny_ready = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// GStreamer callback
GstFlowReturn on_new_sample(GstAppSink *appsink, gpointer user_data) {
    PipelineData *data = (PipelineData*)user_data;
    GstSample *sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return GST_FLOW_ERROR;
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }
    GstCaps *caps = gst_sample_get_caps(sample);
    GstStructure *s = gst_caps_get_structure(caps, 0);
    int width, height;
    gst_structure_get_int(s, "width", &width);
    gst_structure_get_int(s, "height", &height);

    cv::Mat bgr(height, width, CV_8UC3, (uchar*)map.data);
    if (bgr.empty()) {
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    // Pass frame to processing thread
    {
        std::unique_lock<std::timed_mutex> lock(data->frame_mutex, std::chrono::milliseconds(1));
        data->latest_frame = bgr.clone();
    }

    // Get canny mask from thread (non-blocking)
    cv::Mat canny;
    {
        std::lock_guard<std::mutex> lock(canny_mutex);
        if (canny_ready) {
            canny = latest_canny.clone();
            canny_ready = false;
        }
    }

    // Highlight in red using canny mask (main thread)
    cv::Mat output_frame = bgr.clone();
    if (!canny.empty())
        output_frame.setTo(cv::Scalar(0,0,255), canny);

    // Push BGR frame to appsrc (GStreamer will convert to NV12)
    GstBuffer *out_buffer = gst_buffer_new_allocate(NULL, output_frame.total() * output_frame.elemSize(), NULL);
    GstMapInfo out_map;
    gst_buffer_map(out_buffer, &out_map, GST_MAP_WRITE);
    memcpy(out_map.data, output_frame.data, output_frame.total() * output_frame.elemSize());
    gst_buffer_unmap(out_buffer, &out_map);

    GstFlowReturn ret = gst_app_src_push_buffer(data->appsrc, out_buffer);

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    return ret;
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);
    GMainLoop *loop = g_main_loop_new(NULL, FALSE);

    // Input pipeline: qtiqmmfsrc -> videoconvert -> capsfilter (NV12) -> videoconvert_to_bgr -> queue -> appsink (BGR)
    GstElement *input_pipeline = gst_pipeline_new("input-pipeline");
    GstElement *src = gst_element_factory_make("qtiqmmfsrc", "src");
    GstElement *videoconvert_in = gst_element_factory_make("videoconvert", "videoconvert_in");
    GstElement *capsfilter = gst_element_factory_make("capsfilter", "capsfilter");
    GstElement *videoconvert_to_bgr = gst_element_factory_make("videoconvert", "videoconvert_to_bgr");
    GstElement *appsink = gst_element_factory_make("appsink", "appsink");
    GstElement *queue_in = gst_element_factory_make("queue", "queue_in");

    // Output pipeline: appsrc (BGR) -> queue -> videoconvert_to_nv12 -> capsfilter_nv12 -> qtic2venc (NV12) -> h264parse -> rtph264pay -> udpsink
    GstElement *output_pipeline = gst_pipeline_new("output-pipeline");
    GstElement *appsrc = gst_element_factory_make("appsrc", "appsrc");
    GstElement *queue_out = gst_element_factory_make("queue", "queue_out");
    GstElement *videoconvert_to_nv12 = gst_element_factory_make("videoconvert", "videoconvert_to_nv12");
    GstElement *capsfilter_nv12 = gst_element_factory_make("capsfilter", "capsfilter_nv12");
    GstElement *encoder = gst_element_factory_make("qtic2venc", "encoder");
    GstElement *parser = gst_element_factory_make("h264parse", "parser");
    GstElement *payloader = gst_element_factory_make("rtph264pay", "payloader");
    GstElement *udpsink = gst_element_factory_make("udpsink", "udpsink");

    // Set camera and caps
    g_object_set(src, "camera", 1, NULL); // 0 or 1 for camera index
    GstCaps *caps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "NV12",
        "width", G_TYPE_INT, 1280,
        "height", G_TYPE_INT, 720,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        NULL);
    g_object_set(capsfilter, "caps", caps, NULL);

    // Set appsink caps to BGR
    GstCaps *appsink_caps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "BGR",
        "width", G_TYPE_INT, 1280,
        "height", G_TYPE_INT, 720,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        NULL);
    g_object_set(appsink, "caps", appsink_caps, NULL);
    g_object_set(appsink, "emit-signals", TRUE, "sync", FALSE, NULL);
    g_object_set(queue_in, "max-size-buffers", 1, "max-size-bytes", 0, "max-size-time", 0, NULL);

    // Set appsrc caps to BGR (you push BGR frames)
    GstCaps *appsrc_caps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "BGR",
        "width", G_TYPE_INT, 1280,
        "height", G_TYPE_INT, 720,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        NULL);
    g_object_set(appsrc, "caps", appsrc_caps, NULL);

    // Set encoder caps to NV12 (GStreamer will convert BGR to NV12)
    GstCaps *encoder_caps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "NV12",
        "width", G_TYPE_INT, 1280,
        "height", G_TYPE_INT, 720,
        "framerate", GST_TYPE_FRACTION, 30, 1,
        NULL);
    g_object_set(capsfilter_nv12, "caps", encoder_caps, NULL);

    g_object_set(udpsink, "host", "192.168.68.125", "port", 5000, "sync", FALSE, NULL);
    g_object_set(queue_out, "max-size-buffers", 1, "max-size-bytes", 0, "max-size-time", 0, NULL);

    // Build pipelines
    gst_bin_add_many(GST_BIN(input_pipeline), src, videoconvert_in, capsfilter, videoconvert_to_bgr, queue_in, appsink, NULL);
    gst_bin_add_many(GST_BIN(output_pipeline), appsrc, queue_out, videoconvert_to_nv12, capsfilter_nv12, encoder, parser, payloader, udpsink, NULL);

    // Link input pipeline
    if (!gst_element_link_many(src, videoconvert_in, capsfilter, videoconvert_to_bgr, queue_in, appsink, NULL)) {
        std::cerr << "Failed to link input pipeline" << std::endl;
        return -1;
    }
    // Link output pipeline
    if (!gst_element_link_many(appsrc, queue_out, videoconvert_to_nv12, capsfilter_nv12, encoder, parser, payloader, udpsink, NULL)) {
        std::cerr << "Failed to link output pipeline" << std::endl;
        return -1;
    }

    PipelineData data;
    data.appsrc = GST_APP_SRC(appsrc);
    GstAppSinkCallbacks callbacks = {NULL, NULL, on_new_sample, NULL};
    gst_app_sink_set_callbacks(GST_APP_SINK(appsink), &callbacks, &data, NULL);

    // Start homography/motion detection thread
    std::thread homography_thread(homography_thread_func, &data);

    // Start pipelines
    gst_element_set_state(input_pipeline, GST_STATE_PLAYING);
    gst_element_set_state(output_pipeline, GST_STATE_PLAYING);

    std::cout << "Pipelines started" << std::endl;
    g_main_loop_run(loop);

    // Cleanup
    running = false;
    homography_thread.join();
    gst_element_set_state(input_pipeline, GST_STATE_NULL);
    gst_element_set_state(output_pipeline, GST_STATE_NULL);
    gst_object_unref(input_pipeline);
    gst_object_unref(output_pipeline);
    g_main_loop_unref(loop);

    return 0;
}

