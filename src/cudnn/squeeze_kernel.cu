

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

void Squeeze::map(void)
{
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE);
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void Squeeze::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Squeeze::forward(bool block)
{
  copy_kernel<<<GET_BLOCKS(outputs[0].volume()), CUDA_NUM_THREADS>>>(
      (float*)outputs[0].data_ptr, (float*)inputs[0].data_ptr, outputs[0].volume());
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_squeeze_cost(Squeeze* sqz)
{
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int i = 0; i < REPEAT_TIMES; i++) {
    copy_kernel<<<GET_BLOCKS(sqz->outputs[0].volume()), CUDA_NUM_THREADS>>>(
        outputPtr, inputPtr, sqz->outputs[0].volume());
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  sqz->runtime = milliseconds / REPEAT_TIMES;
  if (print_cost)
    printf("  measure[Squeeeze]: cost(%.4lf)\n", sqz->runtime);
}
