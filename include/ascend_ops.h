#ifndef SRC_ASCEND_OPS_H_
#define SRC_ASCEND_OPS_H_

#include <ATen/ATen.h>

at::Tensor npu_test_ops(at::Tensor& x, at::Tensor& y);

/*  catlass算子接口  */
__inline__ void npu_test_matmul_add(at::Tensor left, at::Tensor right, at::Tensor output,int m, int n, int k);

#endif // SRC_ASCEND_OPS_H_
