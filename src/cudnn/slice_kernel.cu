

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

void Slice::map(void)
{
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void Slice::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Slice::forward(bool block)
{
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_slice_cost(Slice* slice)
{
  // FIXME: assume the cost is zero for now
  slice->runtime = 0;
}
