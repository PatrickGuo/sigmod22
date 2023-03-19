

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

void TopK::map(void)
{
  // TODO: use cudnn reduce tensor
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputs[0].volume() * sizeof(DATATYPE)));
  checkCUDA(cudaMalloc(&outputs[1].data_ptr, outputs[1].volume() * sizeof(DATATYPE)));
}

void TopK::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
  checkCUDA(cudaFree(outputs[1].data_ptr));
}

void TopK::forward(bool block)
{
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_topk_cost(TopK* topk)
{
  // TODO: use cudnn reduce tensor
  topk->runtime = 0;
}
