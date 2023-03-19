#include "dnndiff/ops.h"
using namespace dnndiff;

// Diff propagation
void OpBase::diff() {
  // all ops (including OP_INPUT/WEIGHT) are set with difference = 1 initially
  float diff = 0;
  for (int i = 0; i < numInputs; i++) {
    if (inputs[i].op.ptr == NULL) diff += 1; // pure input tensor's op.ptr is NULL.
    else diff += inputs[i].op.ptr->difference;
  }
  difference = diff / (float) numInputs;
}
void Mul::diff() {
  difference = inputs[0].op.ptr->difference * inputs[1].op.ptr->difference;
}

/* Pool2D, Activation all have ||Op(x)|| < ||x|| */

void BatchNorm::diff() {
  float scale = 0;
  float volume = inputs[1].volume();
  float* data_arr = (float*)inputs[1].data_ptr;
  for (int i = 0; i < volume; i++) scale += data_arr[i];
  difference = inputs[0].op.ptr->difference * scale / volume;
}

/* Concat: sqrt(diff1^2 + diff2^2 + ... + diffn^2) */
void Concat::diff() {
  float all = 0;
  for (int i = 0; i < numInputs; i++) {
    all += inputs[i].op.ptr->difference * inputs[i].op.ptr->difference;
  }
  difference = std::sqrt(all);
}

void Element::diff() {
  switch (type) {
    case OP_EW_ADD:
    case OP_EW_SUB:
    case OP_EW_MAX:
    case OP_EW_MIN:
      difference = std::max(inputs[0].op.ptr->difference, inputs[1].op.ptr->difference);
      break;
    case OP_EW_MUL:
    case OP_EW_DIV:
      difference = std::sqrt(inputs[0].op.ptr->difference * inputs[1].op.ptr->difference);
      break;
    case OP_EW_EQUAL:
    case OP_EW_GREATER:
    case OP_EW_LESS:
    case OP_PRELU:
      difference = 1;
      break;
    default:
      difference = (inputs[0].op.ptr->difference + inputs[1].op.ptr->difference) / 2;
  }
}

void ElementWiseUnary::diff() {
  switch (type) {
    case OP_CEIL:
    case OP_ROUND:
    case OP_LOGICAL_NOT:
    case OP_LOG:
      difference = inputs[0].op.ptr->difference;
      break;
    case OP_EXP:
      difference = std::exp(0 - inputs[0].op.ptr->difference);
      break;
    case OP_SQRT:
      difference = std::sqrt(inputs[0].op.ptr->difference);
      break;
    default:
      difference = inputs[0].op.ptr->difference;
  }
}