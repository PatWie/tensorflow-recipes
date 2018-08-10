// origin: NVIDIA FLOWNET2 (pytorch)
// see https://github.com/NVIDIA/flownet2-pytorch
#if GOOGLE_CUDA

#define EIGEN_USE_GPU

// #include <helper_cuda.h>
#include <iostream>
#include <algorithm>

#include "tensorflow/core/framework/op.h"
#include "correlation_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"


namespace {

template<unsigned int THREADS, unsigned int THREADS_PER_BLOCK>
__global__ void pad_and_transpose(const float* input, float* rinput, int channels, int height, int width, int pad_size) {
    // n (batch size), c (num of channels), y (height), x (width)
    const int n = blockIdx.x;
    const int y = blockIdx.y;
    const int x = blockIdx.z;
    const int ch_off = threadIdx.x;

    float value;

    const int dimcyx = channels * height * width;
    const int dimyx = height * width;

    const int p_dimx = (width + 2 * pad_size);
    const int p_dimy = (height + 2 * pad_size);
    const int p_dimyxc = channels * p_dimy * p_dimx;
    const int p_dimxc = p_dimx * channels;

    for (int c = ch_off; c < channels; c += THREADS_PER_BLOCK) {
        value = input[n * dimcyx + c * dimyx + y * width + x];
        rinput[n * p_dimyxc + (y + pad_size) * p_dimxc + (x + pad_size) * channels + c] = value;
    }
}

template<unsigned int THREADS, unsigned int THREADS_PER_BLOCK>
__global__ void Correlation_forward( float *output, int nOutputChannels, int outputHeight, int outputWidth,
                                     float *rInput1, int nInputChannels, int inputHeight, int inputWidth,
                                     float *rInput2,
                                     int pad_size,
                                     int kernel_size,
                                     int max_displacement,
                                     int stride1,
                                     int stride2) {
    // n (batch size), c (num of channels), y (height), x (width)

    const int pInputWidth = inputWidth + 2 * pad_size;
    const int pInputHeight = inputHeight + 2 * pad_size;

    const int kernel_rad = (kernel_size - 1) / 2;
    const int displacement_rad = max_displacement / stride2;
    const int displacement_size = 2 * displacement_rad + 1;

    const int n  = blockIdx.x;
    const int y1 = blockIdx.y * stride1 + max_displacement + kernel_rad;
    const int x1 = blockIdx.z * stride1 + max_displacement + kernel_rad;
    const int c = threadIdx.x;

    const int pdimyxc = pInputHeight * pInputWidth * nInputChannels;
    const int pdimxc = pInputWidth * nInputChannels;
    const int pdimc = nInputChannels;

    const int tdimcyx = nOutputChannels * outputHeight * outputWidth;
    const int tdimyx = outputHeight * outputWidth;
    const int tdimx = outputWidth;

    float nelems = kernel_size * kernel_size * pdimc;

    __shared__ float prod_sum[THREADS_PER_BLOCK];

    // no significant speed-up in using chip memory for input1 sub-data,
    // not enough chip memory size to accomodate memory per block for input2 sub-data
    // instead i've used device memory for both

    // element-wise product along channel axis
    for (int tj = -displacement_rad; tj <= displacement_rad; ++tj ) {
        for (int ti = -displacement_rad; ti <= displacement_rad; ++ti ) {
            prod_sum[c] = 0;
            int x2 = x1 + ti * stride2;
            int y2 = y1 + tj * stride2;

            for (int j = -kernel_rad; j <= kernel_rad; ++j) {
                for (int i = -kernel_rad; i <= kernel_rad; ++i) {
                    for (int ch = c; ch < pdimc; ch += THREADS_PER_BLOCK) {
                        int indx1 = n * pdimyxc + (y1 + j) * pdimxc + (x1 + i) * pdimc + ch;
                        int indx2 = n * pdimyxc + (y2 + j) * pdimxc + (x2 + i) * pdimc + ch;

                        prod_sum[c] += rInput1[indx1] * rInput2[indx2];
                    }
                }
            }

            // accumulate
            __syncthreads();
            if (c == 0) {
                float reduce_sum = 0;
                for (int index = 0; index < THREADS_PER_BLOCK; ++index) {
                    reduce_sum += prod_sum[index];
                }
                int tc = (tj + displacement_rad) * displacement_size + (ti + displacement_rad);
                const int tindx = n * tdimcyx + tc * tdimyx + blockIdx.y * tdimx + blockIdx.z;
                output[tindx] = reduce_sum / nelems;
            }

        }
    }

}

template<unsigned int THREADS, unsigned int THREADS_PER_BLOCK>
__global__ void Correlation_backward_input1(int item, float *gradInput1, int nInputChannels, int inputHeight, int inputWidth,
        const float *gradOutput, int nOutputChannels, int outputHeight, int outputWidth,
        const float *rInput2,
        int pad_size,
        int kernel_size,
        int max_displacement,
        int stride1,
        int stride2) {
    // n (batch size), c (num of channels), y (height), x (width)

    int n = item;
    int y = blockIdx.x * stride1 + pad_size;
    int x = blockIdx.y * stride1 + pad_size;
    int c = blockIdx.z;
    int tch_off = threadIdx.x;

    int kernel_rad = (kernel_size - 1) / 2;
    int displacement_rad = max_displacement / stride2;
    int displacement_size = 2 * displacement_rad + 1;

    int xmin = (x - kernel_rad - max_displacement) / stride1;
    int ymin = (y - kernel_rad - max_displacement) / stride1;

    int xmax = (x + kernel_rad - max_displacement) / stride1;
    int ymax = (y + kernel_rad - max_displacement) / stride1;

    if (xmax < 0 || ymax < 0 || xmin >= outputWidth || ymin >= outputHeight) {
        // assumes gradInput1 is pre-allocated and zero filled
        return;
    }

    if (xmin > xmax || ymin > ymax) {
        // assumes gradInput1 is pre-allocated and zero filled
        return;
    }

    xmin = max(0, xmin);
    xmax = min(outputWidth - 1, xmax);

    ymin = max(0, ymin);
    ymax = min(outputHeight - 1, ymax);

    int pInputWidth = inputWidth + 2 * pad_size;
    int pInputHeight = inputHeight + 2 * pad_size;

    int pdimyxc = pInputHeight * pInputWidth * nInputChannels;
    int pdimxc = pInputWidth * nInputChannels;
    int pdimc = nInputChannels;

    int tdimcyx = nOutputChannels * outputHeight * outputWidth;
    int tdimyx = outputHeight * outputWidth;
    int tdimx = outputWidth;

    int odimcyx = nInputChannels * inputHeight * inputWidth;
    int odimyx = inputHeight * inputWidth;
    int odimx = inputWidth;

    float nelems = kernel_size * kernel_size * nInputChannels;

    __shared__ float prod_sum[THREADS_PER_BLOCK];
    prod_sum[tch_off] = 0;

    for (int tc = tch_off; tc < nOutputChannels; tc += THREADS_PER_BLOCK) {

        int i2 = (tc % displacement_size - displacement_rad) * stride2;
        int j2 = (tc / displacement_size - displacement_rad) * stride2;

        int indx2 = n * pdimyxc + (y + j2) * pdimxc + (x + i2) * pdimc + c;

        float val2 = rInput2[indx2];

        for (int j = ymin; j <= ymax; ++j) {
            for (int i = xmin; i <= xmax; ++i) {
                int tindx = n * tdimcyx + tc * tdimyx + j * tdimx + i;
                prod_sum[tch_off] += gradOutput[tindx] * val2;
            }
        }
    }
    __syncthreads();

    if (tch_off == 0) {
        float reduce_sum = 0;
        for (int idx = 0; idx < THREADS_PER_BLOCK; idx++) {
            reduce_sum += prod_sum[idx];
        }
        const int indx1 = n * odimcyx + c * odimyx + (y - pad_size) * odimx + (x - pad_size);
        gradInput1[indx1] = reduce_sum / nelems;
    }

}

template<unsigned int THREADS, unsigned int THREADS_PER_BLOCK>
__global__ void Correlation_backward_input2(int item, float *gradInput2, int nInputChannels, int inputHeight, int inputWidth,
        const float *gradOutput, int nOutputChannels, int outputHeight, int outputWidth,
        const float *rInput1,
        int pad_size,
        int kernel_size,
        int max_displacement,
        int stride1,
        int stride2) {
    // n (batch size), c (num of channels), y (height), x (width)

    int n = item;
    int y = blockIdx.x * stride1 + pad_size;
    int x = blockIdx.y * stride1 + pad_size;
    int c = blockIdx.z;

    int tch_off = threadIdx.x;

    int kernel_rad = (kernel_size - 1) / 2;
    int displacement_rad = max_displacement / stride2;
    int displacement_size = 2 * displacement_rad + 1;

    int pInputWidth = inputWidth + 2 * pad_size;
    int pInputHeight = inputHeight + 2 * pad_size;

    int pdimyxc = pInputHeight * pInputWidth * nInputChannels;
    int pdimxc = pInputWidth * nInputChannels;
    int pdimc = nInputChannels;

    int tdimcyx = nOutputChannels * outputHeight * outputWidth;
    int tdimyx = outputHeight * outputWidth;
    int tdimx = outputWidth;

    int odimcyx = nInputChannels * inputHeight * inputWidth;
    int odimyx = inputHeight * inputWidth;
    int odimx = inputWidth;

    float nelems = kernel_size * kernel_size * nInputChannels;

    __shared__ float prod_sum[THREADS_PER_BLOCK];
    prod_sum[tch_off] = 0;

    for (int tc = tch_off; tc < nOutputChannels; tc += THREADS_PER_BLOCK) {
        int i2 = (tc % displacement_size - displacement_rad) * stride2;
        int j2 = (tc / displacement_size - displacement_rad) * stride2;

        int xmin = (x - kernel_rad - max_displacement - i2) / stride1;
        int ymin = (y - kernel_rad - max_displacement - j2) / stride1;

        int xmax = (x + kernel_rad - max_displacement - i2) / stride1;
        int ymax = (y + kernel_rad - max_displacement - j2) / stride1;

        if (xmax < 0 || ymax < 0 || xmin >= outputWidth || ymin >= outputHeight) {
            // assumes gradInput2 is pre-allocated and zero filled
            continue;
        }

        if (xmin > xmax || ymin > ymax) {
            // assumes gradInput2 is pre-allocated and zero filled
            continue;
        }

        xmin = max(0, xmin);
        xmax = min(outputWidth - 1, xmax);

        ymin = max(0, ymin);
        ymax = min(outputHeight - 1, ymax);

        int indx1 = n * pdimyxc + (y - j2) * pdimxc + (x - i2) * pdimc + c;
        float val1 = rInput1[indx1];

        for (int j = ymin; j <= ymax; ++j) {
            for (int i = xmin; i <= xmax; ++i) {
                int tindx = n * tdimcyx + tc * tdimyx + j * tdimx + i;
                prod_sum[tch_off] += gradOutput[tindx] * val1;
            }
        }
    }

    __syncthreads();

    if (tch_off == 0) {
        float reduce_sum = 0;
        for (int idx = 0; idx < THREADS_PER_BLOCK; idx++) {
            reduce_sum += prod_sum[idx];
        }
        const int indx2 = n * odimcyx + c * odimyx + (y - pad_size) * odimx + (x - pad_size);
        gradInput2[indx2] = reduce_sum / nelems;
    }

}

// --------------------------------------------------------------
} // anonymouse namespace

namespace tensorflow {
namespace functor {
typedef Eigen::GpuDevice GPUDevice;

template <typename Dtype>
struct CorrelationFunctor<GPUDevice, Dtype> {
    void operator ()(::tensorflow::OpKernelContext* ctx,
                     const Tensor& input_a_t, const Tensor& input_b_t,
                     Tensor *padded_a_t, Tensor *padded_b_t,
                     Tensor *output_t,
                     correlation::parameter *params) {

        enum {
            THREADS = 1024,
            THREADS_PER_BLOCK = 32
        };

        const int B = output_t->dim_size(0);

        const int iC = input_a_t.dim_size(1);
        const int iW = input_a_t.dim_size(3);
        const int iH = input_a_t.dim_size(2);

        const int oC = output_t->dim_size(1);
        const int oW = output_t->dim_size(3);
        const int oH = output_t->dim_size(2);

        dim3 blocks_grid(B, iH, iW);
        dim3 threads_block(THREADS_PER_BLOCK);

        cudaMemset(padded_a_t->flat<Dtype>().data(), 0, padded_a_t->flat<Dtype>().size() * sizeof(Dtype));
        cudaMemset(padded_b_t->flat<Dtype>().data(), 0, padded_b_t->flat<Dtype>().size() * sizeof(Dtype));
        cudaMemset(output_t->flat<Dtype>().data(), 0, output_t->flat<Dtype>().size() * sizeof(Dtype));

        pad_and_transpose<THREADS, THREADS_PER_BLOCK> <<< blocks_grid, threads_block>>>
        (
            input_a_t.flat<Dtype>().data(), padded_a_t->flat<Dtype>().data(),
            iC, iH, iW, params->pad
        );
        pad_and_transpose<THREADS, THREADS_PER_BLOCK> <<< blocks_grid, threads_block>>>
        (
            input_b_t.flat<Dtype>().data(), padded_b_t->flat<Dtype>().data(),
            iC, iH, iW, params->pad
        );

        dim3 threadsPerBlock(THREADS_PER_BLOCK);
        dim3 totalBlocksCorr(B, oH, oW);

        Correlation_forward<THREADS, THREADS_PER_BLOCK> <<< totalBlocksCorr, threadsPerBlock>>>
        (
            output_t->flat<Dtype>().data(), oC, oH, oW,
            padded_a_t->flat<Dtype>().data(), iC, iH, iW,
            padded_b_t->flat<Dtype>().data(),
            params->pad,
            params->kernel_size,
            params->max_displacement,
            params->stride_1,
            params->stride_2
        );

        if (!ctx->eigen_gpu_device().ok()) {
            ctx->SetStatus(tensorflow::errors::Internal("Correlation_forward_cuda_kernel::kernel execution failed on GPU"));
        }

    }
};


template <typename Dtype>
struct CorrelationGradFunctor<GPUDevice, Dtype> {
    void operator ()(::tensorflow::OpKernelContext* ctx,
                     const Tensor& input_a_t, const Tensor& input_b_t,
                     Tensor *padded_a_t, Tensor *padded_b_t,
                     const Tensor &gradients_t,
                     Tensor *output_a_gradient, Tensor *output_b_gradient,
                     correlation::parameter *params) {

        enum {
            THREADS = 1024,
            THREADS_PER_BLOCK = 32
        };

        cudaMemset(padded_a_t->flat<Dtype>().data(), 0, padded_a_t->flat<Dtype>().size() * sizeof(Dtype));
        cudaMemset(padded_b_t->flat<Dtype>().data(), 0, padded_b_t->flat<Dtype>().size() * sizeof(Dtype));
        cudaMemset(output_a_gradient->flat<Dtype>().data(), 0, output_a_gradient->flat<Dtype>().size() * sizeof(Dtype));
        cudaMemset(output_b_gradient->flat<Dtype>().data(), 0, output_b_gradient->flat<Dtype>().size() * sizeof(Dtype));


        int B = gradients_t.dim_size(0);

        int iC = input_a_t.dim_size(1);
        int iH = input_a_t.dim_size(2);
        int iW = input_a_t.dim_size(3);

        int oC = gradients_t.dim_size(1);
        int oH = gradients_t.dim_size(2);
        int oW = gradients_t.dim_size(3);

        dim3 blocks_grid(B, iH, iW);
        dim3 threads_block(THREADS_PER_BLOCK);

        pad_and_transpose<THREADS, THREADS_PER_BLOCK> <<< blocks_grid, threads_block>>>
        (
            input_a_t.flat<Dtype>().data(), padded_a_t->flat<Dtype>().data(),
            iC, iH, iW, params->pad
        );
        pad_and_transpose<THREADS, THREADS_PER_BLOCK> <<< blocks_grid, threads_block>>>
        (
            input_b_t.flat<Dtype>().data(), padded_b_t->flat<Dtype>().data(),
            iC, iH, iW, params->pad
        );

        dim3 threadsPerBlock(THREADS_PER_BLOCK);
        dim3 totalBlocksCorr(iH, iW, iC);

        for (int n = 0; n < B; ++n) {
            Correlation_backward_input1<THREADS, THREADS_PER_BLOCK> << <totalBlocksCorr, threadsPerBlock>> > (
                        n, output_a_gradient->flat<Dtype>().data(), iC, iH, iW,
                        gradients_t.flat<Dtype>().data(), oC, oH, oW,
                        padded_b_t->flat<Dtype>().data(),
                        params->pad,
                        params->kernel_size,
                        params->max_displacement,
                        params->stride_1,
                        params->stride_2
                    );
        }

        for (int n = 0; n < B; n++) {
            Correlation_backward_input2<THREADS, THREADS_PER_BLOCK> <<< totalBlocksCorr, threadsPerBlock>>>(
                n, output_b_gradient->flat<Dtype>().data(), iC, iH, iW,
                gradients_t.flat<Dtype>().data(), oC, oH, oW,
                padded_a_t->flat<Dtype>().data(),
                params->pad,
                params->kernel_size,
                params->max_displacement,
                params->stride_1,
                params->stride_2
            );
        }

        if (!ctx->eigen_gpu_device().ok()) {
            ctx->SetStatus(tensorflow::errors::Internal("Correlation_backward_cuda_kernel::kernel execution failed on GPU"));
        }
    }
};


template struct CorrelationFunctor<GPUDevice, float>;
template struct CorrelationGradFunctor<GPUDevice, float>;

} // namespace functor
} // namespace tensorflow


#endif  // GOOGLE_CUDA
