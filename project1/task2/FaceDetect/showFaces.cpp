/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Computer Vision

  Simple example of face detection using a Haar cascade
*/
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "faceDetect.h"

int main(int argc, char *argv[]) {
  cv::VideoCapture *capdev;

  // open the video device
  capdev = new cv::VideoCapture(0);
  if( !capdev->isOpened() ) {
    printf("Unable to open video device\n");
    return(-1);
  }

  // check the size of the video stream
  cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
		 (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

  printf("Expected size: %d %d\n", refS.width, refS.height);

  cv::namedWindow("Video", 1); // identifies a window?

  cv::Mat frame;
  cv::Mat grey;
  std::vector<cv::Rect> faces;
  cv::Rect last(0, 0, 0, 0);

  // Loop forever
  for(int f=0;;f++) {

    // get a new frame from the camera, treat as a stream
    *capdev >> frame; 

    if( frame.empty() ) {
      printf("frame is empty\n");
      break;
    }

    // convert the image to greyscale
    cv::cvtColor( frame, grey, cv::COLOR_BGR2GRAY, 0);

    // detect faces
    detectFaces( grey, faces );

    // draw boxes around the faces
    drawBoxes( frame, faces );

    // add a little smoothing by averaging the last two detections
    if( faces.size() > 0 ) {
      last.x = (faces[0].x + last.x)/2;
      last.y = (faces[0].y + last.y)/2;
      last.width = (faces[0].width + last.width)/2;
      last.height = (faces[0].height + last.height)/2;
    }

    // display the frame with the box in it
    cv::imshow("Video", frame);
    
    // check for a "q" key
    char key = cv::waitKey(10);
    if( key == 'q') {
      break;
    }
  }

  // terminate the video capture
  printf("Terminating\n");
  delete capdev;

  return(0);
}
