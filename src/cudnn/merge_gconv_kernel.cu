

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

__global__
void merge_gconv_kernel(DATATYPE* dst_ptr,
                        const DATATYPE* src_ptr,
                        int volume,
                        int c_in_h_w,
                        int c_out,
                        int count)
{
  assert(c_out % count == 0);
  CUDA_KERNEL_LOOP(i, volume)
  {
    int mod = i % c_in_h_w;
    int div = i / c_in_h_w;
    int dst_i = div * c_in_h_w * count + div / (c_out / count) * c_in_h_w + mod;
    dst_ptr[dst_i] = src_ptr[i];
  }
}

void MergeGConv::map(void)
{
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void MergeGConv::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void MergeGConv::forward(bool block)
{
  int c_out = inputs[0].dim[0];
  int c_in_h_w = inputs[0].volume() / c_out;
  assert(outputs[0].dim[1] % inputs[0].dim[1] == 0);
  int count = outputs[0].dim[1] / inputs[0].dim[1];
  assign_kernel<<<GET_BLOCKS(outputs[0].volume()), CUDA_NUM_THREADS>>>(
      (DATATYPE*)outputs[0].data_ptr, outputs[0].volume(), 0.0f);
  merge_gconv_kernel<<<GET_BLOCKS(inputs[0].volume()), CUDA_NUM_THREADS>>>(
      (DATATYPE*)outputs[0].data_ptr, (DATATYPE*)inputs[0].data_ptr,
      inputs[0].volume(), c_in_h_w, c_out, count);

  if (block)
    checkCUDA(cudaDeviceSynchronize());
}
