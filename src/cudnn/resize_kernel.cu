

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

void Resize::map(void)
{
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void Resize::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Resize::forward(bool block)
{
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_resize_cost(Resize* resize)
{
  // FIXME: assume the cost is zero for now
  resize->runtime = 0;
}
