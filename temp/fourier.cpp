/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Tutorial 2

  Gaussian blur
*/

#include <cstdio>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include "opencv2/opencv.hpp"  // the top include file

/*
  A useful function for visualiing the important frequencies in an image.

  Computes the log of the magnitude of the Discrete Fourier Transform of an image.

  cv::Mat fft is the original DFT of an image

  cv::Mat mag holds the image for visualization on output, it will be of type CV_32F
 */
int visPowerSpectrum( cv::Mat &fft, cv::Mat &mag ) {
  
  // in order to visualize the spectrum, we compute the magnitude of the complex number and take the log
  mag.create(fft.size(), CV_32F );

  // compute the magnitude and the log
  for(int i=0;i<fft.rows;i++) {
    float *data = fft.ptr<float>(i);
    float *mptr = mag.ptr<float>(i);
    for(int j=0;j<fft.cols;j++) {
      float x = data[j*2];
      float y = data[j*2 + 1];
      mptr[j] = log( 1 + sqrt(x*x + y*y) );
    }
  }

  cv::normalize( mag, mag, 0, 1, cv::NORM_MINMAX );

  // reorganize the quadrants to be centered on the middle of the image
  int cx = mag.cols/2;
  int cy = mag.rows/2;

  cv::Mat q0(mag, cv::Rect( 0, 0, cx, cy ) ); // x, y, width, height
  cv::Mat q1(mag, cv::Rect( cx, 0, cx, cy ) );
  cv::Mat q2(mag, cv::Rect( 0, cy, cx, cy ) );
  cv::Mat q3(mag, cv::Rect( cx, cy, cx, cy ) );

  cv::Mat tmp;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);

  return(0);
}


int buildGaussianFilterImage( double sigma, int rows, int cols, cv::Mat &gaussFilter ) {
  // make a Gaussian filter image
  double gscale = 1.0 / (2 * M_PI * sigma * sigma);
  double sigscale = 2 * sigma * sigma;
  int cx = sigma * 3;
  int cy = sigma * 3;

  gaussFilter = cv::Mat::zeros( cv::Size( cols, rows ), CV_32F );

  // The Gaussian needs to be centered on (0, 0) or it will shift the image
  for(int i=0;i<rows;i++) {
    float *data = gaussFilter.ptr<float>(i);
    for(int j=0;j<cols;j++) {
      double dx = j < cx ? j : j > cols - cx ? cols - j : 1e+7;
      double dy = i < cy ? i : i > rows - cy ? rows - i : 1e+7;
      data[j] = gscale * exp( - (dx*dx / sigscale + dy*dy / sigscale) );
    }
  }

  return(0);
}


// returns a double which gives time in seconds
double getTime() {
  struct timeval cur;

  gettimeofday( &cur, NULL );
  return( cur.tv_sec + cur.tv_usec / 1000000.0 );
}


// reading an image (path on command line), modifying it
int main( int argc, char *argv[] ) {
  cv::Mat src; // standard image data type
  cv::Mat grey; // greyscale version of the image
  cv::Mat paddedGrey; 
  char filename[256];

  // check if enough command line arguments
  if(argc < 2) {
    printf("usage: %s <image filename>\n", argv[0] );
    exit(-1);
  }
  strcpy( filename, argv[1] );

  // read the image
  src = cv::imread( filename ); // allocates the image data, reads as a BGR 8-bit per channel image

  // test of the image read was successful
  if( src.data == NULL ) { // no image data read from file
    printf("error: unable to read image %s\n", filename );
    exit(-1);
  }

  // convert the image to greyscale (easier to deal with 1-D data )
  cv::cvtColor( src, grey, cv::COLOR_BGR2GRAY );

  cv::namedWindow( filename, 1 );
  cv::imshow( filename, grey );

  // The fast DFT wants images to be nicely sized
  int m = cv::getOptimalDFTSize( grey.rows );
  int n = cv::getOptimalDFTSize( grey.cols );
  printf("Resizing image to %d x %d with zero padding\n", m, n);
  cv::copyMakeBorder( grey, paddedGrey, 0, m - grey.rows, n - grey.cols, cv::BORDER_CONSTANT, 0 );

  cv::namedWindow( "padded", 2 );
  cv::imshow( "padded", paddedGrey );

  // make a 2-channel 32-bit float array of images, with zeros for the second band
  cv::Mat planes[] = { cv::Mat_<float>(paddedGrey), cv::Mat::zeros(paddedGrey.size(), CV_32F ) };
  cv::Mat complex;
  cv::Mat fft;
  cv::merge( planes, 2, complex );

  // take the discrete Fourier transform of the image
  cv::dft( complex, fft ); 

  // in order to visualize the spectrum, we compute the magnitude of the complex number and take the log
  cv::Mat mag;
  visPowerSpectrum( fft, mag );
  cv::imshow("DFT2", mag );
  
  double sigma = 10.0; // sigma for the Gaussian
  cv::Mat gaussFilter;
  char key = 0;

  // make a Gaussian filter image
  buildGaussianFilterImage( sigma, src.rows, src.cols, gaussFilter );
  
  
  // start a loop here
  for(;key != 'q';) {

    // get any keypresses
    key = cv::waitKey(10);

    // quit on a q
    switch(key) {
    case 'q':
      break;
    case '+': // make the filter bigger if + or = (same key)
    case '=':
      sigma += 1.0;
      buildGaussianFilterImage( sigma, src.rows, src.cols, gaussFilter );
      break;
    case '-': // make the filter smaller if -
      sigma = sigma > 1.0 ? sigma-1 : sigma;
      buildGaussianFilterImage( sigma, src.rows, src.cols, gaussFilter );
      break;
    default:
      break;
    }
    

    // visualize the filter
    cv::Mat vis;
    gaussFilter.copyTo(vis);
    cv::normalize( vis, vis, 0, 1, cv::NORM_MINMAX );
    cv::imshow("Gauss Filter", vis);

  
    // take the dft of the Gaussian image
    cv::Mat gplanes[] = { cv::Mat_<float>(gaussFilter), cv::Mat::zeros(gaussFilter.size(), CV_32F ) };
    cv::Mat gcomp;
    cv::merge( gplanes, 2, gcomp );

//    cv::dft( gcomp, gcomp );

    // visualize the Gaussian filter FT
    visPowerSpectrum( gcomp, mag );
    cv::imshow("Gauss DFT", mag );

    // multiply the two images
    cv::Mat product;
    product.create( fft.size(), fft.type() );
    cv::mulSpectrums( fft, gcomp, product, 0 ); // use this function b/c the data format can be complicated

    // visualize the product spectrum
    visPowerSpectrum( product, mag );
    cv::imshow( "product FFT", mag );
  
    // take the inverse DFT
    cv::Mat filtered;
    cv::idft( product, filtered, cv::DFT_SCALE ); // dft scale sets it back to the original intensity

    cv::split( filtered, planes );
    cv::Mat result = cv::Mat_<unsigned char>( planes[0] );

    cv::imshow( "Filtered Real", result );
  }

  printf("Terminating\n");

  return(0);
}
  
