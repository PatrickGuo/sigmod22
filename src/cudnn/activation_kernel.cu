

#include "dnndiff/ops.h"
#include "dnndiff/cuda_helper.h"
using namespace dnndiff;

void Activation::map(void)
{
  // create descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  helperSetTensorDescriptor(inputs[0], inputTensor);
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
  cudnnActivationMode_t mode;
  switch (type) {
    case OP_RELU:
    case OP_LEAKYRELU:
      mode = CUDNN_ACTIVATION_RELU;
      break;
    case OP_SIGMOID:
      mode = CUDNN_ACTIVATION_SIGMOID;
      break;
    case OP_TANH:
      mode = CUDNN_ACTIVATION_TANH;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnSetActivationDescriptor(actiDesc, mode,
      CUDNN_NOT_PROPAGATE_NAN, 0.0));
  if (!inPlace) {
    size_t outputSize = sizeof(DATATYPE);
    for (int i = 0; i < inputs[0].numDim; i++)
      outputSize *= inputs[0].dim[i];
    checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
  } else {
    outputs[0].data_ptr = inputs[0].data_ptr;
  }
}

void Activation::unmap(void)
{
  checkCUDNN(cudnnDestroyTensorDescriptor(inputTensor));
  checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
  if (!inPlace) {
    checkCUDA(cudaFree(outputs[0].data_ptr));
  }
}

void Activation::forward(bool block)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  checkCUDNN(cudnnActivationForward(model->dnn, actiDesc,
      &alpha, inputTensor, inputs[0].data_ptr,
      &beta, inputTensor, outputs[0].data_ptr));
  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_activation_cost(Activation* act)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  helperSetTensorDescriptor(act->inputs[0], inputTensor);
  cudnnActivationMode_t mode;
  switch (act->type) {
    case OP_RELU:
    case OP_LEAKYRELU:
      mode = CUDNN_ACTIVATION_RELU;
      break;
    case OP_SIGMOID:
      mode = CUDNN_ACTIVATION_SIGMOID;
      break;
    case OP_TANH:
      mode = CUDNN_ACTIVATION_TANH;
      break;
    default:
      assert(false);
  }
  checkCUDNN(cudnnSetActivationDescriptor(actiDesc, mode,
      CUDNN_NOT_PROPAGATE_NAN, 0.0));
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int i = 0; i < REPEAT_TIMES; i++) {
    if (act->inPlace) {
      checkCUDNN(cudnnActivationForward(dnn, actiDesc,
          &alpha, inputTensor, inputPtr,
          &beta, inputTensor, inputPtr));
    } else {
      checkCUDNN(cudnnActivationForward(dnn, actiDesc,
          &alpha, inputTensor, inputPtr,
          &beta, inputTensor, outputPtr));
    }
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  act->runtime = milliseconds / REPEAT_TIMES;
  if (print_cost)
    printf("  measure[Activation]: i(%d %d %d %d) type(%d) cost(%.4lf)\n",
           act->inputs[0].dim[0], act->inputs[0].dim[1], act->inputs[0].dim[2],
           act->inputs[0].dim[3], act->type, act->runtime);
}

