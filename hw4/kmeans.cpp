/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330

  Implementation of a K-means algorithm
*/


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "kmeans.h"

/*
  data: a std::vector of pixels
  means: a std:vector of means, will contain the cluster means when the function returns
  labels: an allocated array of type int, the same size as the data, contains the labels when the function returns
  K: the number of clusters
  maxIterations: maximum number of E-M interactions, default is 10
  stopThresh: if the means change less than the threshold, the E-M loop terminates, default is 0

  Executes K-means clustering on the data
 */
int kmeans( std::vector<cv::Vec3b> &data, std::vector<cv::Vec3b> &means, int *labels, int K, int maxIterations, int stopThresh ) {

  // error checking
  if( K > data.size() ) {
    printf("error: K must be less than the number of data points\n");
    return(-1);
  }

  // clear the means vector
  means.clear();

  // initialize the K mean values
  // use comb sampling to select K values
  int delta = data.size() / K;
  int istep = rand() % (data.size() % K);
  for(int i=0;i<K;i++) {
    int index = (istep + i*delta) % data.size();
    means.push_back( data[index] );
  }
  // have K initial means

  // loop the E-M steps
  for(int i=0;i<maxIterations;i++) {

    // classify each data point using SSD
    for(int j=0;j<data.size();j++) {
      int minssd = SSD( means[0], data[j] );
      int minidx = 0;
      for(int k=1;k<K;k++) {
	int tssd = SSD( means[k], data[j] );
	if( tssd < minssd ) {
	  minssd = tssd;
	  minidx = k;
	}
      }
      labels[j] = minidx;
    }

    // calculate the new means
    std::vector<cv::Vec4i> tmeans(means.size(), cv::Vec4i(0, 0, 0, 0) ); // initialize with zeros
    for(int j=0;j<data.size();j++) {
      tmeans[ labels[j] ][0] += data[j][0];
      tmeans[ labels[j] ][1] += data[j][1];
      tmeans[ labels[j] ][2] += data[j][2];
      tmeans[ labels[j] ][3] ++; // counter
    }
    
    int sum = 0;
    for(int k=0;k<tmeans.size();k++) {
      tmeans[k][0] /= tmeans[k][3];
      tmeans[k][1] /= tmeans[k][3];
      tmeans[k][2] /= tmeans[k][3];

      // compute the SSD between the new and old means
      sum += SSD( tmeans[k], means[k] );

      means[k][0] = tmeans[k][0]; // update the mean
      means[k][1] = tmeans[k][1]; // update the mean
      means[k][2] = tmeans[k][2]; // update the mean
    }

    // check if we can stop early
    if( sum <= stopThresh ) {
      break;
    }
  }

  // the labels and updated means are the final values

  return(0);
}
