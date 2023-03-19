

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

void Pad::map(void)
{
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputs[0].volume() * sizeof(DATATYPE)));
}

void Pad::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Pad::forward(bool block)
{
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_pad_cost(Pad* pad)
{
  pad->runtime = 0;
}
