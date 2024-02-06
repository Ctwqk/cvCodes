/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330

  Execute k-means on the colors of an image

  Display the image using the colors chosen by the k-means algorithm
*/


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>
#include "kmeans.h"

/*
  Given a pixel and an array of pixels, returns the closest color in the array
 */
cv::Vec3b closestColor( cv::Vec3b &pix, std::vector<cv::Vec3b> &colors ) {
  int mincolor = 0;
  int mindist = SSD( pix, colors[0] );

  for(int i=1;i<colors.size();i++) {
    int sse = SSD( pix, colors[i] );
    if( sse < mindist ) {
      mindist = sse;
      mincolor = i;
    }
  }

  return( colors[mincolor] );
}

// Main function looks for a command line argument with the image filename and # of colors
int main(int argc, char *argv[]) {

  cv::Mat src;
  cv::Mat dst;  // viewing image
  char filename[256];
  int ncolors = 16;

  // error checking
  if(argc < 3) {
    printf("usage: %s <image filename> <# of colors>\n", argv[0]);
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
	cv::namedWindow("Original",cv::WINDOW_NORMAL);
	cv::resizeWindow("Original",800,600);
  cv::imshow("Original", src );

  // grab the numbers of colors from the command line
  int tcolors = atoi( argv[2] );
  if( tcolors < 1 || tcolors > 66000 ) {
    printf("error: number of colors must be in [1, 66000]\n");
  }
  else {
    ncolors = tcolors;
  }
	//std::cout<<ncolors<<std::endl;	
  // sample colors from the image using jitter sampling
  // sample one color from each B x B block of the image
  int B = 4;
	
  std::vector<cv::Vec3b> data;
  for(int i=0;i<src.rows - B;i += B) {
    for(int j=0;j<src.cols - B;j += B) {
      int jx = rand() % B;
      int jy = rand() % B;
      data.push_back( src.at<cv::Vec3b>(i+jy, j+jx) );
    }
  }

  std::vector<cv::Vec3b> means;
  int *labels = new int[data.size()];
  if(kmeans( data, means, labels, ncolors ) ) {
    printf("Erro using kmeans\n");
    return(-1);
  }
  // kmeans should give a set of means (ncolors of them)
  dst.create( src.size(), src.type() );
  for(int i=0;i<src.rows;i++) {
    for(int j=0;j<src.cols;j++) {
      dst.at<cv::Vec3b>(i,j) = closestColor( src.at<cv::Vec3b>(i,j), means );
    }
  }

	cv::namedWindow("clustered",cv::WINDOW_NORMAL);
	cv::resizeWindow("clustered",800,600);
  cv::imshow("clustered", dst );

  cv::waitKey(0);

  delete[] labels;

  return(0);
}
