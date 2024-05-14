#include "dnnEmbedding.h"
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
