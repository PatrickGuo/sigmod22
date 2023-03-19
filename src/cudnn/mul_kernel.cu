

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

void Mul::map(void)
{
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void Mul::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Mul::forward(bool block)
{
  assert(false);
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_mul_cost(Mul* m)
{
  assert(false);
}
