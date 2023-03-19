#include "dnndiff/ops.h"
using namespace dnndiff;

TensorHandle Graph::leakyrelu(const TensorHandle _input, float alpha, bool _inPlace)
{
  Op op = model->get_or_create_activation(*_input, OP_LEAKYRELU, _inPlace);
  assert(op != Op::INVALID_OP);
  add_edge(_input->op, op, _input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

TensorHandle Graph::relu(const TensorHandle _input, bool _inPlace)
{
  Op op = model->get_or_create_activation(*_input, OP_RELU, _inPlace);
  assert(op != Op::INVALID_OP);
  add_edge(_input->op, op, _input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

TensorHandle Graph::sigmoid(const TensorHandle _input, bool _inPlace)
{
  Op op = model->get_or_create_activation(*_input, OP_SIGMOID, _inPlace);
  assert(op != Op::INVALID_OP);
  add_edge(_input->op, op, _input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

TensorHandle Graph::tanh(const TensorHandle _input, bool _inPlace)
{
  Op op = model->get_or_create_activation(*_input, OP_TANH, _inPlace);
  assert(op != Op::INVALID_OP);
  add_edge(_input->op, op, _input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_activation(Tensor _input, OpType _type, bool _inPlace)
{
  // keys are (inputN, inputC, inputH, inputW, _type, _inPlace)
  ActivationKey key(_input, _type, _inPlace);
  Activation* actOp;
  if (activation.find(key) != activation.end()) {
    actOp = activation[key];
  } else {
    actOp = new Activation(this, _input, _type, _inPlace);
    measure_activation_cost(actOp);
    activation[key] = actOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = actOp;
  return ret;
}

Activation::Activation(Model* _model, Tensor _input, OpType _type, bool _inPlace)
: OpBase(_input, _model, _type), inPlace(_inPlace)
{
  numOutputs = 1;
  outputs[0] = _input;
  outputs[0].idx = 0;
}

Activation::~Activation(void)
{
}

bool Activation::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

void Activation::collect_costs(float& exe_time, float& flops,
                               float& mem_acc, int& num_kernels)
{
  int outputSize = 1, inputSize = 1;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  for (int i = 0; i < inputs[0].numDim; i++)
    inputSize *= inputs[0].dim[i];
  // cost metrics
  exe_time += runtime;
  if (type == OP_RELU)
    flops += 0; // relu does not involve flops
  else
    flops += outputSize;
  mem_acc += inputSize;
  num_kernels += 1;
  printf("        cost[Activation]: mode(%d) cost(%.4lf) total_cost(%.4lf)\n",
         type, runtime, exe_time);
}

// Key ordering: type, inPlace, _input
ActivationKey::ActivationKey(Tensor _input, OpType _type, bool _inPlace)
{
  int idx = 0;
  keys[idx++] = _type;
  keys[idx++] = (int)(_inPlace);
  _input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}

