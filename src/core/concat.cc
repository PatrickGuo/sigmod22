#include "dnndiff/ops.h"
using namespace dnndiff;

TensorHandle Graph::concat(int axis, int n, const TensorHandle* _inputs)
{
  Tensor inputTensors[MAX_NUM_INPUTS];
  for (int i = 0; i < n; i++) {
    inputTensors[i] = *_inputs[i];
  }
  bool needCopy[MAX_NUM_INPUTS];
  for (int i = 0; i < n; i++)
    needCopy[i] = true;
  Op op = model->get_or_create_concat(axis, n, inputTensors, needCopy);
  // Assert op must be valid
  assert (op != Op::INVALID_OP);
  for (int i = 0; i < n; i++) 
    add_edge(_inputs[i]->op, op, _inputs[i]->idx, i);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_concat(int axis, int n, Tensor* _inputs, bool* _needCopy)
{
  // key ordering is:
  // axis, n, bitmask(needCopy)
  // inputs[0].dim[0], ..., inputs[0].dim[axis-1],
  // inputs[0].dim[axis+1], ..., inputs[0].dim[nDims - 1]
  // inputs[0].dim[axis], ..., inputs[n-1].dim[axis]
  // Check validness
  for (int i = 0; i < n; i++) {
    if (_inputs[i].numDim != _inputs[0].numDim) {
      return Op::INVALID_OP;
    }
    for (int j = 0; j < _inputs[0].numDim; j++)
      if ((j != axis) && (_inputs[i].dim[j] != _inputs[0].dim[j])) {
        return Op::INVALID_OP;
      }
  }
  ConcatKey key(axis, n, _inputs, _needCopy);
  Concat* concatOp;
  if (concat.find(key) != concat.end()) {
    concatOp = concat[key];
  } else {
    concatOp = new Concat(this, axis, n, _inputs, _needCopy);
    measure_concat_cost(concatOp);
    concat[key] = concatOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = concatOp;
  return ret;
}

Concat::Concat(Model* _model, int _axis, int n, Tensor* _inputs, bool* _needCopy)
  : OpBase(n, _inputs, _model, OP_CONCAT), axis(_axis)
{
  //for (int i = 0; i < n; i++) {
  //  printf("  concat2[%d]:", i);
  //  for (int j = 0; j < _inputs[i].numDim; j++)
  //    printf("%d, ", _inputs[i].dim[j]);
  //  printf("\n");
  //}
  assert(n <= MAX_NUM_INPUTS);
  for (int i = 0; i < n; i++)
    needCopy[i] = _needCopy[i];
  numOutputs = 1;
  outputs[0].numDim = inputs[0].numDim;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputs[0].dim[i] = inputs[0].dim[i];
  for (int i = 0; i < outputs[0].numDim; i++)
    if (i != axis) {
      outputs[0].split[i] = inputs[0].split[i];
      for (int j = 1; j < n; j++)
        outputs[0].split[i].combine(inputs[j].split[i]);
    }
  outputs[0].split[axis] = inputs[0].split[axis];
  for (int i = 1; i < n; i++) {
    outputs[0].split[axis].merge(outputs[0].dim[axis], inputs[i].split[axis]);
    outputs[0].dim[axis] += inputs[i].dim[axis];
  }
  for (int i = outputs[0].numDim-1; i >= 0; i--) {
    if (i == outputs[0].numDim-1)
      outputs[0].stride[i] = 1;
    else
      outputs[0].stride[i] = outputs[0].stride[i+1] * outputs[0].dim[i+1];
  }
  outputs[0].idx = 0;
}

Concat::~Concat(void)
{}

bool Concat::get_int_parameter(PMParameter para, int* value)
{
  switch (para) {
    case PM_AXIS:
      *value = axis;
      return true;
    default:
      return OpBase::get_int_parameter(para, value);
  }
}

void Concat::collect_costs(float& exe_time, float& flops,
                           float& mem_acc, int& num_kernels)
{
  for (int i = 0; i < numInputs; i++)
    if (needCopy[i]) {
      int inputSize = 1;
      for (int j = 0; j < inputs[i].numDim; j++)
        inputSize *= inputs[i].dim[j];
      mem_acc += inputSize;
    }
  // cost metrics
  exe_time += runtime;
  flops += 0;
  num_kernels += 1;
  printf("        cost[Concat]: numInputs(%d) cost(%.4lf) total_cost(%.4lf)\n",
         numInputs, runtime, exe_time);
}

int bitmask(int n, bool* bits)
{
  int ret = 0;
  for (int i = 0; i < n; i++)
    ret = bits[i] ? ret * 2 + 1 : ret * 2;
  return ret;
}

// key ordering is: axis, n, bitmask(needCopy), inputs[0], ..., inputs[n-1]
//
//
// axis, n, bitmask(needCopy), inputs[0], inputs[n-1]
// inputs[0].dim[0], ..., inputs[0].dim[axis-1],
// inputs[0].dim[axis+1], ..., inputs[0].dim[nDims - 1]
// inputs[0].dim[axis], ..., inputs[n-1].dim[axis]
ConcatKey::ConcatKey(int axis, int n, Tensor* _inputs, bool* _needCopy)
{
  int idx = 0;
  keys[idx++] = axis;
  keys[idx++] = n;
  keys[idx++] = bitmask(n, _needCopy);
  for (int i = 0; i < n; i++)
    _inputs[i].serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
#ifdef DEADCODE
  assert(_inputs[0].numDim + n + 2 <= KEY_LENGTH);
  int idx = 0;
  keys[idx++] = axis;
  keys[idx++] = n;
  keys[idx++] = bitmask(n, _needCopy);
  for (int i = 0; i < _inputs[0].numDim; i++)
    if (i != axis)
      keys[idx++] = _inputs[0].dim[i];
  for (int i = 0; i < n; i++)
    keys[idx++] = _inputs[i].dim[axis];
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
#endif
}

