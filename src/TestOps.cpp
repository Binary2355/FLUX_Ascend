#include "OpApiCommon.h"
#include "ascend_ops.h"

at::Tensor npu_test_ops(at::Tensor& x, at::Tensor& y)
{
    EXEC_NPU_CMD(aclnnTestOps, x, y);
    return x;
}
