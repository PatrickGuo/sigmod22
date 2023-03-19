

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

__global__
void enlarge_kernel(DATATYPE* dst_ptr,
                    const DATATYPE* src_ptr,
                    int volume,
                    int dst_h,
                    int dst_w,
                    int src_h,
                    int src_w)
{
  int off_h = (dst_h - src_h) / 2;
  int off_w = (dst_w - src_w) / 2;
  CUDA_KERNEL_LOOP(i, volume)
  {
    int h = (i % (dst_h * dst_w)) / dst_w - off_h;
    int w = (i % (dst_h * dst_w)) % dst_w - off_w;
    if ((h < 0) || (h >= src_h) || (w < 0) || (w >= src_w))
      dst_ptr[i] = 0.0f;
    else {
      int offset = (i / (dst_h * dst_w)) * (src_h * src_w) + h * src_w + w;
      dst_ptr[i] = src_ptr[offset];
    }
  }
}

void Enlarge::map(void)
{
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void Enlarge::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Enlarge::forward(bool block)
{
  enlarge_kernel<<<GET_BLOCKS(outputs[0].volume()), CUDA_NUM_THREADS>>>(
      (DATATYPE*)outputs[0].data_ptr, (DATATYPE*)inputs[0].data_ptr, outputs[0].volume(),
      outputs[0].dim[2], outputs[0].dim[3], inputs[0].dim[2], inputs[0].dim[3]);
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_enlarge_cost(Enlarge* enl)
{
  enl->runtime = 0.0f;
}
