

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

bool ElementWiseUnary::use_kernel(void) const
{
  return false;
}

void ElementWiseUnary::map(void)
{
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputs[0].volume() * sizeof(DATATYPE)));
}

void ElementWiseUnary::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void ElementWiseUnary::forward(bool block)
{
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_elementwise_unary_cost(ElementWiseUnary* unary)
{
  unary->runtime = 0;
  if (print_cost)
    printf("  measure[ElementWiseUnary]: type(%d) cost(%.4lf)\n",
           unary->type, unary->runtime);
}
