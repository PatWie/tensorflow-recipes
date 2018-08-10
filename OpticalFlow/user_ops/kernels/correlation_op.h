/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#ifndef CORRELATION_OP_HH
#define CORRELATION_OP_HH


#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
class OpKernelContext;
class Tensor;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
}


namespace tensorflow {
namespace functor {

namespace correlation {
struct parameter {
 public:
  int inB;
  int inC;
  int inH;
  int inW;


  int kernel_size;
  int max_displacement;
  int stride_1;
  int stride_2;
  int pad;

  int radius;
  int border;

  parameter(int B_, int C_, int W_, int H_) {
    inB = B_;
    inC = C_;
    inH = H_;
    inW = W_;
  }

  parameter(const Tensor& intput,
            int kernel_size_, int max_displacement_,
            int stride_1_, int stride_2_,
            int pad_) {
    inB = intput.dim_size(0);
    inC = intput.dim_size(1);
    inH = intput.dim_size(2);
    inW = intput.dim_size(3);

    kernel_size = kernel_size_;
    max_displacement = max_displacement_;
    stride_1 = stride_1_;
    stride_2 = stride_2_;
    pad = pad_;
    update();
  }

  void update() {
    radius = (kernel_size - 1) / 2;
    border = max_displacement + radius;
  }

  int B() const { return inB;}
  int C() const {
    const int n_radius = max_displacement / stride_2;
    const int n_width  = n_radius * 2 + 1;
    return n_width * n_width;
  }
  int H() const {
    int pH = inH + 2 * pad;
    return static_cast<int>(
             ceil(static_cast<float>((pH - border * 2)) /
                  static_cast<float>(stride_1)));
  }
  int W() const {
    int pW = inW + 2 * pad;
    return static_cast<int>(
             ceil(static_cast<float>((pW - border * 2)) /
                  static_cast<float>(stride_1)));
  }
};
};  // namespace correlation


template <typename Device, typename Dtype>
struct CorrelationFunctor {
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const Tensor& input_a_t, const Tensor& input_b_t,
                   Tensor *padded_a_t, Tensor *padded_b_t,
                   Tensor *output_t,
                   correlation::parameter *params);
};

template <typename Device, typename Dtype>
struct CorrelationGradFunctor {
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const Tensor& input_a_t, const Tensor& input_b_t,
                   Tensor *padded_a_t, Tensor *padded_b_t,
                   const Tensor &gradients_t,
                   Tensor *output_a_gradient, Tensor *output_b_gradient,
                   correlation::parameter *params);
};

}  // namespace functor
}  // namespace tensorflow


#endif  // TENSORFLOW_USER_OPS_KERNELS_CORRELATION_OP_H_
