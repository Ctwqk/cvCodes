/*
  Bruce A. Maxwell
  CS 5330 Spring 2024
  Project 3

  Reads a 2DOR network and an image and applies the image to the network to generate an embedding

  Also prints out the model layers.  The fc1/Gemm layer is used as the embedding

  The getEmbedding function can be used with a real-time 2D OR system to obtain a feature vector for a 2D object
*/
#include <cstdio>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include "opencv2/opencv.hpp"  // the top include file
#include "opencv2/dnn.hpp"


/*
  cv::Mat src        thresholded and cleaned up image in 8UC1 format
  cv::Mat ebmedding  holds the embedding vector after the function returns
  cv::Rect bbox      the axis-oriented bounding box around the region to be identified
  cv::dnn::Net net   the pre-trained network
  int debug          1: show the image given to the network and print the embedding, 0: don't show extra info
 */
int getEmbedding( cv::Mat &src, cv::Mat &embedding, cv::Rect &bbox, cv::dnn::Net &net, int debug=0 );
int getEmbedding( cv::Mat &src, cv::Mat &embedding, cv::Rect &bbox, cv::dnn::Net &net, int debug ) {
  const int ORNet_size = 128;
  cv::Mat padImg;
  cv::Mat blob;
	
  cv::Mat roiImg = src( bbox );
  int top = bbox.height > 128 ? 10 : (128 - bbox.height)/2 + 10;
  int left = bbox.width > 128 ? 10 : (128 - bbox.width)/2 + 10;
  int bottom = top;
  int right = left;
	
  cv::copyMakeBorder( roiImg, padImg, top, bottom, left, right, cv::BORDER_CONSTANT, 0  );
  cv::resize( padImg, padImg, cv::Size( 128, 128 ) );

  cv::dnn::blobFromImage( src, // input image
			  blob, // output array
			  (1.0/255.0) / 0.5, // scale factor
			  cv::Size( ORNet_size, ORNet_size ), // resize the image to this
			  128,   // subtract mean prior to scaling
			  false, // input is a single channel image
			  true,  // center crop after scaling short side to size
			  CV_32F ); // output depth/type

  net.setInput( blob );
  embedding = net.forward( "onnx_node!/fc1/Gemm" );

  if(debug) {
    cv::imshow( "pad image", padImg );
    std::cout << embedding << std::endl;
    cv::waitKey(0);
  }

  return(0);
}


/*
  Main function

  Expects two command line arguments: <network path> <image path>

  The image should be a thresholded image with one object in it.
 */
int main(int argc, char *argv[]) {
  char mod_filename[256];
  char img_filename[256];

  if(argc < 3) {
    printf("usage: %s <network path> <image path>\n", argv[0] );
    exit(-1);
  }

  strncpy(mod_filename, argv[1], 255);
  strncpy(img_filename, argv[2], 255);

  // read the network
  cv::dnn::Net net = cv::dnn::readNet( mod_filename );
  printf("Network read successfully\n");

  /// print the names of the layers
  std::vector<cv::String> names = net.getLayerNames();

  for(int i=0;i<names.size();i++) {
    printf("Layer %d: '%s'\n", i, names[i].c_str() );
  }

  // read image and convert it to greyscale
  cv::Mat src = cv::imread( img_filename );
  cv::cvtColor( src, src, cv::COLOR_BGR2GRAY );

  // the getEmbedding function wants a rectangle that contains the object to be recognized
  cv::Rect bbox( 0, 0, src.cols, src.rows );

  // get the embedding
  cv::Mat embedding;
  getEmbedding( src, embedding, bbox, net, 1 );  // change the 1 to a 0 to turn off debugging

  return(0);
}
