

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

void Shape::map(void)
{
  // TODO: use cudnn reduce tensor
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputs[0].volume() * sizeof(DATATYPE)));
}

void Shape::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Shape::forward(bool block)
{
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_shape_cost(Shape* shape)
{
  // TODO: use cudnn reduce tensor
  shape->runtime = 0;
}
