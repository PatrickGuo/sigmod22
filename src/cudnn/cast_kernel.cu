

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

void Cast::map(void)
{
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputs[0].volume() * sizeof(DATATYPE)));
}

void Cast::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Cast::forward(bool block)
{
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_cast_cost(Cast* cast)
{
  cast->runtime = 0;
  if (print_cost)
    printf("  measure[Cast]: type(%d) cost(%.4lf)\n",
           cast->type, cast->runtime);
}

