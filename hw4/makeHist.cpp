/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330

  Generate a histogram, displaa a visualization of it.
*/


#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>

// Main function looks for a command line argument with the image filename
int main(int argc, char *argv[]) {

  cv::Mat src;
  cv::Mat dst;  // viewing image
  cv::Mat hist; // histogram data

  char filename[256];
  float max;
  const int histsize = 256;

  // error checking
  if(argc < 2) {
    printf("usage: %s <image filename>\n", argv[0]);
    return(-1);
  }

  // grab the filename
  strcpy( filename, argv[1] );

  // read the file
  src = cv::imread( filename );  
  if( src.data == NULL ) {
    printf("error: unable to read filename %s\n", filename);
    return(-2);
  }

  // initialize the histogram (use floats so we can make probabilities)
  hist = cv::Mat::zeros( cv::Size( histsize, histsize ), CV_32FC1 );

  // keep track of largest bucket for visualization purposes
  max = 0;

  // loop over all pixels
  for( int i=0;i<src.rows;i++) {
    cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i); // pointer to row i
    for(int j=0;j<src.cols;j++) {

      // get the RGB values
      float B = ptr[j][0];
      float G = ptr[j][1];
      float R = ptr[j][2];

      // compute the r,g chromaticity
      float divisor = R + G + B;
      divisor = divisor > 0.0 ? divisor : 1.0; // check for all zeros
      float r = R / divisor;
      float g = G / divisor;

      // compute indexes, r, g are in [0, 1]
      int rindex = (int)( r * (histsize - 1) + 0.5 );
      int gindex = (int)( g * (histsize - 1) + 0.5 );

      // increment the histogram
      hist.at<float>(rindex, gindex)++;

      // keep track of the size of the largest bucket (just so we know what it is)
      float newvalue = hist.at<float>(rindex, gindex);
      max = newvalue > max ? newvalue : max;
    }
  }

  // print information
  printf("The largest bucket has %d pixels in it\n", (int)max );
  
  // histogram is complete
  // normalize the histogram by the number of pixels
  hist /= (src.rows * src.cols); // divides all elements of a cv::Mat by a scalar

  // the chromaticity histogram is ready for saving as a feature vector now

  // make a visualization of a histogram
  // color each bin by its chromaticity and weight
  dst.create( hist.size(), CV_8UC3 );
  for(int i=0;i<hist.rows;i++) {
    cv::Vec3b *ptr = dst.ptr<cv::Vec3b>(i);
    float *hptr = hist.ptr<float>(i);
    for(int j=0;j<hist.cols;j++) {
      if( i + j > hist.rows ) {
	ptr[j] = cv::Vec3b( 200, 120, 60 ); // default color
	continue;
      }
      
      float rcolor = (float)i / histsize;
      float gcolor = (float)j / histsize;
      float bcolor = 1 - (rcolor + gcolor);

      ptr[j][0] = hptr[j] > 0 ? hptr[j] * 128 + 128 * bcolor : 0 ;
      ptr[j][1] = hptr[j] > 0 ? hptr[j] * 128 + 128 * gcolor : 0 ;
      ptr[j][2] = hptr[j] > 0 ? hptr[j] * 128 + 128 * rcolor : 0 ;
    }
  }
  
  // whole histogram is in the range [0, 1]
  cv::imshow( "Histogram", dst );

  // can save the histogram to a file if you want

  cv::waitKey(0);

  return(0);
}
