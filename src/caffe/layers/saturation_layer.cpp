#include <vector>

#include "caffe/layers/saturation_layer.hpp"

namespace caffe {

template <typename Dtype>
void SaturationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  saturation_ = this->layer_param_.saturation_param().saturation();
}

template <typename Dtype>
void SaturationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::min(std::max(bottom_data[i], (-saturation_)), (saturation_));
  }
}

template <typename Dtype>
void SaturationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      if(bottom_data[i] >= (-saturation_) && bottom_data[i] <= saturation_){
          bottom_diff[i] = top_diff[i];
      }
      else{
          bottom_diff[i] = 0;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SaturationLayer);
#endif

INSTANTIATE_CLASS(SaturationLayer);
REGISTER_LAYER_CLASS(Saturation);

}  // namespace caffe    