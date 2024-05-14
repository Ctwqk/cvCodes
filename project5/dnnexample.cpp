/*
  Bruce A. Maxwell
  CS 5330 Spring 2024
  Project 3

  Reads a 2DOR network and an image and applies the image to the network to generate an embedding

  Also prints out the model layers.  The fc1/Gemm layer is used as the embedding

  The getEmbedding function can be used with a real-time 2D OR system to obtain a feature vector for a 2D object
*/
#include "dnnEmbedding.h"
/*
  cv::Mat src        thresholded and cleaned up image in 8UC1 format
  cv::Mat ebmedding  holds the embedding vector after the function returns
  cv::Rect bbox      the axis-oriented bounding box around the region to be identified
  cv::dnn::Net net   the pre-trained network
  int debug          1: show the image given to the network and print the embedding, 0: don't show extra info
 */

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
