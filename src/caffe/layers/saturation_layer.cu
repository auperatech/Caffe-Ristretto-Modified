#include <vector>

#include "caffe/layers/saturation_layer.hpp"

namespace caffe {

    template <typename Dtype>
__global__ void SaturationForward(const int n, const Dtype* in, Dtype* out,
    Dtype saturation) {
        float tmp;
  CUDA_KERNEL_LOOP(index, n) {
    tmp = in[index] > saturation ? saturation : in[index];
    out[index] = tmp < (-saturation) ? (-saturation) : tmp;
  }
}

template <typename Dtype>
void SaturationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SaturationForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, saturation_);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void SaturationBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype saturation) {
  CUDA_KERNEL_LOOP(index, n) {
    if(in_data[index] >= (-saturation) && in_data[index] <= saturation){
        out_diff[index] = in_diff[index];
    }
    else{
        out_diff[index] = 0;
    }
  }
}

template <typename Dtype>
void SaturationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SaturationBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, saturation_);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SaturationLayer);


}  // namespace caffe