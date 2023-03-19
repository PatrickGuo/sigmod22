#include "dnndiff/ops.h"
using namespace dnndiff;

TensorHandle Graph::shape(const TensorHandle _input,
                          OpType _type)
{
  Op op = model->get_or_create_shape(*_input, _type);
  assert(op != Op::INVALID_OP);
  add_edge(_input->op, op, _input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_shape(const Tensor& _input,
                              OpType _type)
{
  ShapeKey key(_input, _type);
  Shape* shapeOp;
  if (shape.find(key) != shape.end()) {
    shapeOp = shape[key];
  } else {
    shapeOp = new Shape(this, _input, _type);
    measure_shape_cost(shapeOp);
    shape[key] = shapeOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = shapeOp;
  return ret;
}

Shape::Shape(Model* _model, const Tensor& _input, OpType _type)
: OpBase(_input, _model, _type)
{
  numOutputs = 1;
  outputs[0].numDim = 1;
  if (type == OP_SHAPE) {
    outputs[0].dim[0] = _input.numDim;
    outputs[0].split[0] = SplitInfo::NO_SPLIT;
    outputs[0].stride[0] = 1;
  } else {
    assert(type == OP_SIZE);
    outputs[0].dim[0] = 1;
    outputs[0].split[0] = SplitInfo::NO_SPLIT;
    outputs[0].stride[0] = 1;
  }
  outputs[0].idx = 0;
}

Shape::~Shape(void)
{}

bool Shape::get_int_parameter(PMParameter para, int* value)
{
  return OpBase::get_int_parameter(para, value);
}

void Shape::collect_costs(float& exe_time, float& flops,
                          float& mem_acc, int& num_kernels)
{
  exe_time += runtime;
  num_kernels += 1;
  printf("      cost[Shape]: cost(%.4lf) total_cost(%.4lf)\n",
         runtime, exe_time);
}

ShapeKey::ShapeKey(const Tensor& _input, OpType _type)
{
  int idx = 0;
  keys[idx++] = _type;
  _input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}
