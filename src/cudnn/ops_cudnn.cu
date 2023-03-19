

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"

using namespace dnndiff;

Model::Model()
: isTraining(false), print_cost(false)
{
  //int* a = (int*) malloc(sizeof(int) * 8);
  checkCUDA(cudaSetDevice(0));
  checkCUDNN(cudnnCreate(&dnn));
  checkCUDA(cublasCreate(&blas));
  checkCUDA(cusolverDnCreate(&solver)); // DeCor : svd
  workSpaceSize = WORK_SPACE_SIZE;
  global_unique_id = 100;
  checkCUDA(cudaMalloc(&workSpace, workSpaceSize));
  // printf("handle.workSpace = 0x%x\n", workSpace);
  // create all descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&scaleTensor));
  checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
  checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
  checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc));
  // allocate tensors for measuring performance
  checkCUDA(cudaMalloc(&inputPtr, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&biasPtr, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&outputPtr, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&filterPtr, MAX_TENSOR_SIZE));
  // create tensors for batch norm
  checkCUDA(cudaMalloc(&scalePtr, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&runningMean, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&runningVar, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&saveMean, MAX_TENSOR_SIZE));
  checkCUDA(cudaMalloc(&saveVar, MAX_TENSOR_SIZE));
  // create cuda events
  checkCUDA(cudaEventCreate(&startEvent));
  checkCUDA(cudaEventCreate(&endEvent));
}

float Model::measure_oplist_runtime(const std::vector<OpBase*>& opBaseList)
{
  const int num_runs = 100;
  // warmup
  for (int times = 0; times < num_runs; times++)
    for (int i = 0; i < opBaseList.size(); i++)
      opBaseList[i]->forward();
  // measure runtime
  // checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int times = 0; times < num_runs; times++) {
    for (int i = 0; i < opBaseList.size(); i++)
      opBaseList[i]->forward();
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  return milliseconds / num_runs;
}

void* Model::allocate_memory(size_t size, const DATATYPE* data_initial)
{
  void* ptr;
  if (size == 0) {
    // Note: Special value for zero-sized tensor
    ptr = (void*) 0x1;
  } else {
    checkCUDA(cudaMalloc(&ptr, size));
  }
  if (data_initial != NULL) {
    checkCUDA(cudaMemcpy(ptr, data_initial, size, cudaMemcpyDefault));
  }
  return ptr;
}

bool Model::copy_memory(DATATYPE* dst, const DATATYPE* src, size_t size)
{
  checkCUDA(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
  return true;
}
