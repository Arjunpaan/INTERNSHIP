#!/bin/bash
g++ -std=c++11 -pthread myfile27.cc -o myfile27 $(pkg-config --cflags --libs opencv4 gstreamer-1.0 gstreamer-app-1.0)


#g++ -std=c++11 myfile1.cc `pkg-config --cflags --libs opencv4 gstreamer-1.0 gstreamer-app-1.0` -o yolov5_fire_detection


#g++ -o my_file1_video myfile1.cc $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 opencv4)
#$(pkg-config --cflags --libs gstreamer-1.0)








