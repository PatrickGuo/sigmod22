#include "dnndiff/ops.h"
using namespace dnndiff;

TensorHandle Graph::input_wrapper(const TensorHandle _input)
{
  // Always create new operator for input
  Op op = model->create_input(*_input, OP_INPUT);
  add_edge(_input->op, op, _input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

TensorHandle Graph::weight_wrapper(const TensorHandle _weight)
{
  // Always create new operator for weight
  Op op = model->create_weight(*_weight, OP_WEIGHT);
  add_edge(_weight->op, op, _weight->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

// TODO: we ignore dropout rate for inference
TensorHandle Graph::dropout(const TensorHandle _input)
{
  Op op = model->get_or_create_noop(*_input, OP_DROPOUT);
  add_edge(_input->op, op, _input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::create_input(Tensor _input, OpType _type)
{
  assert(_type == OP_INPUT);
  Op ret;
  ret.ptr = new NoOp(this, _input, _type);
  ret.guid = global_unique_id ++;
  return ret;
}

Op Model::create_weight(Tensor _weight, OpType _type)
{
  assert(_type == OP_WEIGHT);
  //assert(_weight.data_ptr != NULL);
  if (_weight.data_ptr == NULL) {
    fprintf(stderr, "[%s:%d] Warning: Find uninitialized weight tensor.\n", __FILE__, __LINE__);
  }
  Op ret;
  ret.ptr = new NoOp(this, _weight, _type);
  ret.guid = global_unique_id ++;
  return ret;
}

Op Model::get_or_create_noop(Tensor _input, OpType _type)
{
  assert(_type == OP_DROPOUT);
  // key is (_type, _input)
  NoopKey key(_input, _type);
  NoOp* noOp;
  if (noop.find(key) != noop.end()) {
    noOp = noop[key];
  } else {
    noOp = new NoOp(this, _input, _type);
    noOp->runtime = 0.0f;
    noop[key] = noOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = noOp;
  return ret;
}

NoOp::NoOp(Model* _model, Tensor _input, OpType type)
: OpBase(_input, _model, type)
{
  numOutputs = 1;
  outputs[0] = _input;
  outputs[0].idx = 0;
}

NoOp::~NoOp(void)
{}

bool NoOp::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

void NoOp::map(void)
{}

void NoOp::unmap(void)
{}

void NoOp::forward(bool block)
{}

void NoOp::collect_costs(float& exe_time, float& flops,
                         float& mem_acc, int& num_kernels)
{
  // cost metrics
  exe_time += 0;
  flops += 0;
  mem_acc += 0;
  num_kernels += 0;
}

// key ordering: _type, input
NoopKey::NoopKey(Tensor input, OpType _type)
{
  int idx = 0;
  keys[idx++] = _type;
  input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}

