#include "boost/algorithm/string.hpp"

#include "caffe/caffe.hpp"
#include "ristretto/quantize_CHaidnn.hpp"


using caffe::Caffe;
using caffe::Net;
using caffe::string;
using caffe::vector;
using caffe::Blob;
using caffe::LayerParameter;
using caffe::NetParameter;

Quantize_CHaidnn::Quantize_CHaidnn(string model, string weights, string model_quantized,
      int iterations, string trimming_mode, double error_margin, string gpus) {
  this->model_ = model;
  this->weights_ = weights;
  this->model_quantized_ = model_quantized;
  this->iterations_ = iterations;
  this->trimming_mode_ = trimming_mode;
  this->error_margin_ = error_margin;
  this->gpus_ = gpus;
}

