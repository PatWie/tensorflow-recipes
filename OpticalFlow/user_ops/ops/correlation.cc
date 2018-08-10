// #include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "../kernels/correlation_op.h"

namespace tensorflow {
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;


REGISTER_OP("Correlation")
.Attr("T: realnumbertype")
.Input("input_a: T")
.Input("input_b: T")
.Attr("kernel_size: int")
.Attr("max_displacement: int")
.Attr("stride_1: int")
.Attr("stride_2: int")
.Attr("pad: int")
.Output("output: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  ShapeHandle input_a, input_b, input;

  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_a));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &input_b));
  TF_RETURN_IF_ERROR(c->Merge(input_a, input_b, &input));

  const int B = c->Value(c->Dim(input, 0));
  const int C = c->Value(c->Dim(input, 1));
  const int H = c->Value(c->Dim(input, 2));
  const int W = c->Value(c->Dim(input, 3));

  ::tensorflow::functor::correlation::parameter params(B, C, H, W);
  TF_RETURN_IF_ERROR(c->GetAttr("kernel_size", &params.kernel_size));
  TF_RETURN_IF_ERROR(c->GetAttr("max_displacement", &params.max_displacement));
  TF_RETURN_IF_ERROR(c->GetAttr("stride_1", &params.stride_1));
  TF_RETURN_IF_ERROR(c->GetAttr("stride_2", &params.stride_2));
  TF_RETURN_IF_ERROR(c->GetAttr("pad", &params.pad));
  params.update();

  c->set_output(0, c->MakeShape({params.B(), params.C(), params.H(), params.W()}));
  return Status::OK();
})
.Doc(R"doc(
Apply correlation layer to two inputs.

Computes entire cost volume for two inputs.

input_a: A batch of feature inputs [B, C, H, W].
input_b: A batch of feature inputs [B, C, H, W].
kernel_size: todo.
max_displacement: todo.
stride_1: todo.
stride_2: todo.
pad: todo.
output: Computed cost volume.
)doc");

REGISTER_OP("CorrelationGrad")
.Attr("T: realnumbertype")
.Input("gradients: T")
.Input("input_a: T")
.Input("input_b: T")
.Attr("kernel_size: int")
.Attr("max_displacement: int")
.Attr("stride_1: int")
.Attr("stride_2: int")
.Attr("pad: int")
.Output("bottomdiff_a: T")
.Output("bottomdiff_b: T")
.SetShapeFn([](InferenceContext *c) {
  ShapeHandle shp_hnd;
  TF_RETURN_IF_ERROR(c->Merge(c->input(1), c->input(2), &shp_hnd));
  c->set_output(0, shp_hnd);
  c->set_output(1, shp_hnd);
  return Status::OK();
});
}  // namespace tensorflow
