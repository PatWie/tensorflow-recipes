// Patrick Wieschollek, <mail@patwie.com>, 2018
#include "correlation_cost_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


template <typename Device, typename T>
class CorrelationCostOp : public OpKernel {
 public:
  explicit CorrelationCostOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &kernel_size));
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_displacement", &max_displacement));
    OP_REQUIRES_OK(context, context->GetAttr("stride_1", &stride_1));
    OP_REQUIRES_OK(context, context->GetAttr("stride_2", &stride_2));
    OP_REQUIRES_OK(context, context->GetAttr("pad", &pad));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, kernel_size % 2 != 0,
                errors::InvalidArgument("kernel_size must be odd"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_a_t = context->input(0);
    const Tensor& input_b_t = context->input(1);

    // we didn't check the batch-dimension during "SetShapeFn"
    OP_REQUIRES(context, input_a_t.shape() == input_b_t.shape(),
                errors::InvalidArgument("Input shapes have to be the same"));

    const int32 N = GetTensorDim(input_a_t, data_format_, 'N');
    const int32 H = GetTensorDim(input_a_t, data_format_, 'H');
    const int32 W = GetTensorDim(input_a_t, data_format_, 'W');

    // output channels are d**2 where, d = 2r + 1
    const int32 r = max_displacement / stride_2;
    const int32 d = 2 * r + 1;
    const int32 border = max_displacement + (kernel_size - 1) / 2;

    const int32 Cout = d * d;
    const int32 Hout =
        static_cast<int>(ceil(static_cast<float>(((H + 2 * pad) - border * 2)) /
                              static_cast<float>(stride_1)));
    const int32 Wout =
        static_cast<int>(ceil(static_cast<float>(((W + 2 * pad) - border * 2)) /
                              static_cast<float>(stride_1)));

    OP_REQUIRES(context, Hout >= 1,
                errors::InvalidArgument(
                    "Neighborhood and kernel don't fit in input height."));
    OP_REQUIRES(context, Wout >= 1,
                errors::InvalidArgument(
                    "Neighborhood and kernel don't fit in input width."));

    Tensor* output_t;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({N, Cout, Hout, Wout}),
                                          &output_t));


    functor::CorrelationCostFunctor<Device, T> correlationCostFunc;
    Status s = correlationCostFunc(
        context, input_a_t, input_b_t, output_t,
        /* params */
        kernel_size, max_displacement, stride_1, stride_2, pad, data_format_);

    OP_REQUIRES_OK(context, s);
  }

 private:
  int kernel_size;
  int max_displacement;
  int stride_1;
  int stride_2;
  int pad;
  TensorFormat data_format_;
};

template <typename Device, typename T>
class CorrelationCostGradOp : public OpKernel {
 public:
  explicit CorrelationCostGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &kernel_size));
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_displacement", &max_displacement));
    OP_REQUIRES_OK(context, context->GetAttr("stride_1", &stride_1));
    OP_REQUIRES_OK(context, context->GetAttr("stride_2", &stride_2));
    OP_REQUIRES_OK(context, context->GetAttr("pad", &pad));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, kernel_size % 2 != 0,
                errors::InvalidArgument("kernel_size must be odd"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_a_t = context->input(0);
    const Tensor& input_b_t = context->input(1);
    const Tensor& topdiff_t = context->input(2);

    OP_REQUIRES(context, input_a_t.shape() == input_b_t.shape(),
                errors::InvalidArgument("Input shapes have to be the same"));

    // Allocate the memory for the bottom diffs
    Tensor* output_a_gradient_t;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_a_t.shape(),
                                                     &output_a_gradient_t));
    Tensor* output_b_gradient_t;
    OP_REQUIRES_OK(context, context->allocate_output(1, input_b_t.shape(),
                                                     &output_b_gradient_t));

    functor::CorrelationCostGradFunctor<Device, T> correlationCostGrad;
    Status s = correlationCostGrad(
        context, input_a_t, input_b_t, topdiff_t,
        output_a_gradient_t, output_b_gradient_t,
        /* params */
        kernel_size, max_displacement, stride_1, stride_2, pad, data_format_);

    OP_REQUIRES_OK(context, s);
  }

 private:
  int kernel_size;
  int max_displacement;
  int stride_1;
  int stride_2;
  int pad;
  TensorFormat data_format_;
};

// Register the CPU kernels.
#define REGISTER_CORRELATIONCOST_OP_CPU(T)                                   \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("CorrelationCost").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
      CorrelationCostOp<CPUDevice, T>)                                       \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("CorrelationCostGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CorrelationCostGradOp<CPUDevice, T>)

TF_CALL_float(REGISTER_CORRELATIONCOST_OP_CPU);
TF_CALL_double(REGISTER_CORRELATIONCOST_OP_CPU);
#undef REGISTER_CORRELATIONCOST_OP_CPU

// Register the GPU kernels.
#ifdef GOOGLE_CUDA

#define REGISTER_CORRELATIONCOST_OP_GPU(T)                                   \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("CorrelationCost").Device(DEVICE_GPU).TypeConstraint<T>("T"),     \
      CorrelationCostOp<GPUDevice, T>)                                       \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("CorrelationCostGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      CorrelationCostGradOp<GPUDevice, T>)

TF_CALL_float(REGISTER_CORRELATIONCOST_OP_GPU);
TF_CALL_double(REGISTER_CORRELATIONCOST_OP_GPU);
#undef REGISTER_CORRELATIONCOST_OP_GPU

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
