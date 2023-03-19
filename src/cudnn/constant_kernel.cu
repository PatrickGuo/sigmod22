

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

void Constant::map(void)
{
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void Constant::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Constant::forward(bool block)
{
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

