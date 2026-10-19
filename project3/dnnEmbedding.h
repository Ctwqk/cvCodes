// Embedding of a segmented object from the course-supplied 2D object-recognition
// network (NetWorkThings/or2d-normmodel-007.onnx, 128x128 greyscale input, the
// fc1 layer is used as a 50-value embedding).
#ifndef DNNEMBEDDING_H
#define DNNEMBEDDING_H

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

// Name of the embedding layer in the ONNX graph as OpenCV names it.
extern const char* const kEmbeddingLayer;

/*
  src        8-bit single channel picture (thresholded / cleaned up, object bright on dark)
  embedding  receives the embedding as a 1 x N CV_32F row
  bbox       axis-aligned box around the object (clipped to the image)
  net        the loaded network
  debug      1: show the padded input and print the embedding
  Returns 0, or -1 if the box is empty or the picture has the wrong type.

  Fix compared to the course sample: the network was fed the whole picture instead of the
  padded region of interest it had just prepared.
 */
int getEmbedding(const cv::Mat& src, cv::Mat& embedding, const cv::Rect& bbox, cv::dnn::Net& net, int debug = 0);

// Embedding of one region mask: crops the mask to the bounding box of `box`.
std::vector<float> regionEmbedding(const cv::Mat& mask, const cv::RotatedRect& box, cv::dnn::Net& net);

#endif
