#ifndef QUANTIZE_CHAIDNN_HPP_
#define QUANTIZE_CHAIDNN_HPP_

#include "caffe/caffe.hpp"
#include "ristretto/quantization.hpp"

using caffe::string;
using caffe::vector;
using caffe::Net;

/**
 * @brief Approximate 32-bit floating point networks.
 *
 * This is the Ristretto tool. Use it to generate file descriptions of networks
 * which use reduced word width arithmetic.
 */
class Quantize_CHaidnn : public Quantization {
public:
  explicit Quantize_CHaidnn(string model, string weights, string model_quantized,
      int iterations, string trimming_mode, double error_margin, string gpus);
  
};

#endif // QUANTIZATION_HPP_
