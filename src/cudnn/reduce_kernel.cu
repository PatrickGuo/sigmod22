

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

void Reduce::map(void)
{
  // TODO: use cudnn reduce tensor
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputs[0].volume() * sizeof(DATATYPE)));
}

void Reduce::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Reduce::forward(bool block)
{
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_reduce_cost(Reduce* reduce)
{
  // TODO: use cudnn reduce tensor
  reduce->runtime = 0;
}
