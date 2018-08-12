// Patrick Wieschollek, <mail@patwie.com>, 2018
#ifndef TENSORFLOW_CORRELATION_COST_OP_H_
#define TENSORFLOW_CORRELATION_COST_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct CorrelationCostFunctor {
  Status operator()(OpKernelContext* context, const Tensor& input_a_t,
                    const Tensor& input_b_t, Tensor* output_t,
                    /* params */
                    int kernel_size, int max_displacement, int stride_1,
                    int stride_2, int pad, TensorFormat data_format);
};

template <typename Device, typename T>
struct CorrelationCostGradFunctor {
  Status operator()(OpKernelContext* context, const Tensor& input_a_t,
                    const Tensor& input_b_t, const Tensor& topdiff_t,
                    Tensor* output_a_gradient_t, Tensor* output_b_gradient_t,
                    /* params */
                    int kernel_size, int max_displacement, int stride_1,
                    int stride_2, int pad, TensorFormat data_format);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORRELATION_COST_OP_H_
