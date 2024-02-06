/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330

  Header file for implementation of a K-means algorithm
*/
#ifndef KMEANS_H
#define KMEANS_H

#define SSD(a, b) ( ((int)a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]) )

int kmeans( std::vector<cv::Vec3b> &data, std::vector<cv::Vec3b> &means, int *labels, int K, int maxIterations=10, int stopThresh=0 );


#endif
