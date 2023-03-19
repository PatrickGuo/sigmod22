

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

void Transpose::map(void)
{
  //TODO: for now the output and input share the same instance
  outputs[0].data_ptr = inputs[0].data_ptr;
}

void Transpose::unmap(void)
{
}

void Transpose::forward(bool block)
{
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_transpose_cost(Transpose* transpose)
{
  // Transpose requires no kernel launch
  transpose->runtime = 0;
}
