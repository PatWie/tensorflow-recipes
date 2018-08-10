#define EIGEN_USE_THREADS

#include "correlation_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;


template<typename Device, typename Dtype>
class CorrelationOp : public OpKernel {
 public:
  explicit CorrelationOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &kernel_size));
    OP_REQUIRES_OK(context, context->GetAttr("max_displacement", &max_displacement));
    OP_REQUIRES_OK(context, context->GetAttr("stride_1", &stride_1));
    OP_REQUIRES_OK(context, context->GetAttr("stride_2", &stride_2));
    OP_REQUIRES_OK(context, context->GetAttr("pad", &pad));
    OP_REQUIRES(context, kernel_size % 2 != 0, errors::InvalidArgument("kernel_size must be odd"));
  }

  void Compute(OpKernelContext *context) override {
    const Tensor& input_a_t = context->input(0);
    const Tensor& input_b_t = context->input(1);

    OP_REQUIRES(context, input_a_t.shape() == input_b_t.shape(),
                errors::InvalidArgument("Input shapes have to be the same"));
    OP_REQUIRES(context, input_a_t.dims() == 4, errors::InvalidArgument("input_a_t must have rank 4"));
    OP_REQUIRES(context, input_b_t.dims() == 4, errors::InvalidArgument("input_b_t must have rank 4"));

    ::tensorflow::functor::correlation::parameter params(input_a_t,
        kernel_size, max_displacement, stride_1, stride_2, pad);

    OP_REQUIRES(context, params.H() >= 1,
                errors::InvalidArgument("Neighborhood and kernel don't fit in input height."));
    OP_REQUIRES(context, params.W() >= 1,
                errors::InvalidArgument("Neighborhood and kernel don't fit in input width."));


    Tensor *output_t;
    OP_REQUIRES_OK(context, context->allocate_output(
                     0,
                     TensorShape({params.B(), params.C(), params.H(), params.W()}),
                     &output_t));

    Tensor padded_a_t;
    Tensor padded_b_t;
    TensorShape padded_shape({params.B(),
                              params.inH + 2 * params.pad,
                              params.inW + 2 * params.pad,
                              params.inC
                             });
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value,
                                           padded_shape,
                                           &padded_a_t));
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value,
                                           padded_shape,
                                           &padded_b_t));


    ::tensorflow::functor::CorrelationFunctor<Device, Dtype>()(
      context,
      input_a_t, input_b_t,
      &padded_a_t, &padded_b_t,
      output_t,
      &params);
  }

 private:
  int kernel_size;
  int max_displacement;
  int stride_1;
  int stride_2;
  int pad;
};


template<typename Device, typename Dtype>
class CorrelationGradOp : public OpKernel {
 public:
  explicit CorrelationGradOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &kernel_size));
    OP_REQUIRES_OK(context, context->GetAttr("max_displacement", &max_displacement));
    OP_REQUIRES_OK(context, context->GetAttr("stride_1", &stride_1));
    OP_REQUIRES_OK(context, context->GetAttr("stride_2", &stride_2));
    OP_REQUIRES_OK(context, context->GetAttr("pad", &pad));
    OP_REQUIRES(context, kernel_size % 2 != 0, errors::InvalidArgument("kernel_size must be odd"));
  }

  void Compute(OpKernelContext *context) override {
    const Tensor& gradients_t = context->input(0);
    const Tensor& input_a_t   = context->input(1);
    const Tensor& input_b_t   = context->input(2);

    OP_REQUIRES(context, input_a_t.shape() == input_b_t.shape(),
                errors::InvalidArgument("Input shapes have to be the same"));
    OP_REQUIRES(context, input_a_t.dims() == 4, errors::InvalidArgument("input_a_t must have rank 4"));
    OP_REQUIRES(context, input_b_t.dims() == 4, errors::InvalidArgument("input_b_t must have rank 4"));

    ::tensorflow::functor::correlation::parameter params(input_a_t,
        kernel_size, max_displacement, stride_1, stride_2, pad);

    Tensor padded_a_t;
    Tensor padded_b_t;
    TensorShape padded_shape({params.B(),
                              params.inH + 2 * params.pad,
                              params.inW + 2 * params.pad,
                              params.inC
                             });
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value,
                                           padded_shape,
                                           &padded_a_t));
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<Dtype>::value,
                                           padded_shape,
                                           &padded_b_t));

    // Allocate the memory for the outputs
    Tensor *output_a_gradient_t;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_a_t.shape(), &output_a_gradient_t));
    Tensor *output_b_gradient_t;
    OP_REQUIRES_OK(context, context->allocate_output(1, input_b_t.shape(), &output_b_gradient_t));


    ::tensorflow::functor::CorrelationGradFunctor<Device, Dtype>()(
      context,
      input_a_t, input_b_t,
      &padded_a_t, &padded_b_t,
      gradients_t,
      output_a_gradient_t, output_b_gradient_t,
      &params);


  }

 private:
  int kernel_size;
  int max_displacement;
  int stride_1;
  int stride_2;
  int pad;
};


#define OPNAME(NAME) NAME ## Op
#define REGISTER(NAME, Dtype, DEVICE)                                    \
  REGISTER_KERNEL_BUILDER(                                               \
      Name(#NAME).Device(DEVICE_ ## DEVICE).TypeConstraint<Dtype>("T"),  \
      OPNAME(NAME)<DEVICE ## Device, Dtype>);

REGISTER(Correlation, float, GPU);
REGISTER(CorrelationGrad, float, GPU);


} // end namespace tensorflow
