#include <vector>

#include "caffe/layers/saturation_layer.hpp"

namespace caffe {

template <typename Dtype>
void SaturationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  // saturation_ = this->layer_param_.saturation_param().saturation();
  int bw = this->layer_param_.saturation_param().bit_width();
  int fl = this->layer_param_.saturation_param().fractional_width();

  Dtype pow_bw_minus_1 = pow(2, bw-1);
  pow_minus_fl = pow(2, -fl);
  max_data = (pow_bw_minus_1 - 1) * pow_minus_fl;
  min_data = -pow_bw_minus_1 * pow_minus_fl;

}

template <typename Dtype>
void SaturationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    
    Dtype data = bottom_data[i];
    data = std::max(std::min(data, max_data), min_data);
    data /= pow_minus_fl;
		data = round(data);
		data *= pow_minus_fl;
    top_data[i] = data;
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
      if(bottom_data[i] >= min_data && bottom_data[i] <= max_data){
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