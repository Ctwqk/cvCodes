// Output of the MNIST network for a region of a picture (course sample, adapted).
#ifndef DNNEMBEDDING_H
#define DNNEMBEDDING_H

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

/*
  cv::Mat src        8-bit single channel picture (white digit on black)
  cv::Mat embedding  receives the network output (1x10 log-probabilities)
  cv::Rect bbox      axis-aligned box around the digit; clipped to the picture
  cv::dnn::Net net   the loaded network
  int debug          1: show the padded input and print the output
  Returns 0, or -1 for an empty box / wrong picture type.
 */
int getEmbedding(const cv::Mat& src, cv::Mat& embedding, const cv::Rect& bbox, cv::dnn::Net& net, int debug = 0);

#endif
