

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

void Reshape::map(void)
{
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void Reshape::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Reshape::forward(bool block)
{
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_reshape_cost(Reshape* reshape)
{
  // FIXME: assume the cost is zero for now
  reshape->runtime = 0;
}
