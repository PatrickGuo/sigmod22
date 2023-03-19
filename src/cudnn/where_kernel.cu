

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

void Where::map(void)
{
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputs[0].volume() * sizeof(DATATYPE)));
}

void Where::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Where::forward(bool block)
{
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_where_cost(Where* where)
{
  where->runtime = 0;
}
