// Patrick Wieschollek, <mail@patwie.com>, 2018
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "correlation_cost_op.h"

namespace tensorflow {
namespace functor {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Dtype>
struct CorrelationCostFunctor<CPUDevice, Dtype> {
  Status operator()(OpKernelContext* context, const Tensor& input_a_t,
                    const Tensor& input_b_t, Tensor* output_t,
                    /* params */
                    int kernel_size, int max_displacement, int stride_1,
                    int stride_2, int pad, TensorFormat data_format) {
    const int32 oN = GetTensorDim(*output_t, FORMAT_NCHW, 'N');
    const int32 oH = GetTensorDim(*output_t, FORMAT_NCHW, 'H');
    const int32 oW = GetTensorDim(*output_t, FORMAT_NCHW, 'W');
    const int32 iH = GetTensorDim(input_a_t, data_format, 'H');
    const int32 iW = GetTensorDim(input_a_t, data_format, 'W');
    const int32 iC = GetTensorDim(input_a_t, data_format, 'C');

    const int K = kernel_size * kernel_size * iC;

    const auto input_a = input_a_t.tensor<Dtype, 4>();
    const auto input_b = input_b_t.tensor<Dtype, 4>();
    auto output = output_t->tensor<Dtype, 4>();
    output.setZero();

    const int kernel_rad = (kernel_size - 1) / 2;
    const int displacement_rad = max_displacement / stride_2;
    const int displacement_size = 2 * displacement_rad + 1;

    const bool is_NCHW = (data_format == FORMAT_NCHW);

    for (int n = 0; n < oN; ++n) {
      for (int h = 0; h < oH; ++h) {
        const int h1 = (h - pad) * stride_1 + max_displacement + kernel_rad;
        for (int w = 0; w < oW; ++w) {
          const int w1 = (w - pad) * stride_1 + max_displacement + kernel_rad;

          for (int tj = -displacement_rad; tj <= displacement_rad; ++tj) {
            for (int ti = -displacement_rad; ti <= displacement_rad; ++ti) {
              const int tc = (tj + displacement_rad) * displacement_size +
                             (ti + displacement_rad);

              const int w2 = w1 + ti * stride_2;
              const int h2 = h1 + tj * stride_2;

              for (int j = -kernel_rad; j <= kernel_rad; ++j) {
                // out-of-bound tests
                if ((h1 + j < 0) || (h1 + j >= iH)) continue;
                if ((h2 + j < 0) || (h2 + j >= iH)) continue;
                for (int i = -kernel_rad; i <= kernel_rad; ++i) {
                  if ((w1 + i < 0) || (w1 + i >= iW)) continue;
                  if ((w2 + i < 0) || (w2 + i >= iW)) continue;
                  for (int c = 0; c < iC; ++c) {
                    // eq. (1) in FlowNet: Learning Optical Flow with
                    // Convolutional Networks
                    if (is_NCHW) {
                      output(n, tc, h, w) += input_a(n, c, h1 + j, w1 + i) *
                                             input_b(n, c, h2 + j, w2 + i);
                    } else {
                      output(n, tc, h, w) += input_a(n, h1 + j, w1 + i, c) *
                                             input_b(n, h2 + j, w2 + i, c);
                    }
                  }
                }
              }
              output(n, tc, h, w) /= K;
            }
          }
        }
      }
    }
    return Status::OK();
  }
};

template <typename Dtype>
struct CorrelationCostGradFunctor<CPUDevice, Dtype> {
  Status operator()(OpKernelContext* context, const Tensor& input_a_t,
                    const Tensor& input_b_t, const Tensor& topdiff_t,
                    Tensor* output_a_gradient_t, Tensor* output_b_gradient_t,
                    /* params */
                    int kernel_size, int max_displacement, int stride_1,
                    int stride_2, int pad, TensorFormat data_format) {
    const int32 iN = GetTensorDim(input_a_t, data_format, 'N');
    const int32 iC = GetTensorDim(input_a_t, data_format, 'C');
    const int32 iH = GetTensorDim(input_a_t, data_format, 'H');
    const int32 iW = GetTensorDim(input_a_t, data_format, 'W');

    // topdiff is NCHW
    const int32 oH = GetTensorDim(topdiff_t, FORMAT_NCHW, 'H');
    const int32 oW = GetTensorDim(topdiff_t, FORMAT_NCHW, 'W');

    const auto topdiff = topdiff_t.tensor<Dtype, 4>();
    const auto input_a = input_a_t.tensor<Dtype, 4>();
    const auto input_b = input_b_t.tensor<Dtype, 4>();
    auto output_a_gradient = output_a_gradient_t->tensor<Dtype, 4>();
    auto output_b_gradient = output_b_gradient_t->tensor<Dtype, 4>();
    output_a_gradient.setZero();
    output_b_gradient.setZero();

    const int kernel_rad = (kernel_size - 1) / 2;
    const int displacement_rad = max_displacement / stride_2;
    const int displacement_size = 2 * displacement_rad + 1;
    const int K = kernel_size * kernel_size * iC;

    const bool is_NCHW = (data_format == FORMAT_NCHW);

    for (int n = 0; n < iN; ++n) {
      for (int h = 0; h < oH; ++h) {
        const int h1 = (h - pad) * stride_1 + max_displacement + kernel_rad;
        for (int w = 0; w < oW; ++w) {
          const int w1 = (w - pad) * stride_1 + max_displacement + kernel_rad;

          for (int tj = -displacement_rad; tj <= displacement_rad; ++tj) {
            for (int ti = -displacement_rad; ti <= displacement_rad; ++ti) {
              const int tc = (tj + displacement_rad) * displacement_size +
                             (ti + displacement_rad);

              const int w2 = w1 + ti * stride_2;
              const int h2 = h1 + tj * stride_2;

              for (int j = -kernel_rad; j <= kernel_rad; ++j) {
                // out-of-bound test
                if ((h1 + j < 0) || (h1 + j >= iH)) continue;
                if ((h2 + j < 0) || (h2 + j >= iH)) continue;
                for (int i = -kernel_rad; i <= kernel_rad; ++i) {
                  if ((w1 + i < 0) || (w1 + i >= iW)) continue;
                  if ((w2 + i < 0) || (w2 + i >= iW)) continue;
                  for (int c = 0; c < iC; ++c) {
                    // derivative of eq. (1) in FlowNet
                    if (is_NCHW) {
                      output_a_gradient(n, c, h1 + j, w1 + i) +=
                          topdiff(n, tc, h, w) * input_b(n, c, h2 + j, w2 + i) /
                          K;
                      output_b_gradient(n, c, h2 + j, w2 + i) +=
                          topdiff(n, tc, h, w) * input_a(n, c, h1 + j, w1 + i) /
                          K;
                    } else {
                      output_a_gradient(n, h1 + j, w1 + i, c) +=
                          topdiff(n, tc, h, w) * input_b(n, h2 + j, w2 + i, c) /
                          K;
                      output_b_gradient(n, h2 + j, w2 + i, c) +=
                          topdiff(n, tc, h, w) * input_a(n, h1 + j, w1 + i, c) /
                          K;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return Status::OK();
  }
};

template struct CorrelationCostFunctor<CPUDevice, float>;
template struct CorrelationCostFunctor<CPUDevice, double>;
template struct CorrelationCostGradFunctor<CPUDevice, float>;
template struct CorrelationCostGradFunctor<CPUDevice, double>;

} // namespace functor
} // namespace tensorflow