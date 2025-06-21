#include "OpApiCommon.h"
#include "ascend_ops.h"
#include "test_matmul_add.h"

void npu_test_matmul_add(at::Tensor left, at::Tensor right, at::Tensor output)
{
    if (left.device().type() != c10::DeviceType::PrivateUse1) {
        AT_ERROR("Tensor is not on NPU device!");
    }
    GM_ADDR left_addr = left.data_ptr<float>();
    GM_ADDR right_addr = right.data_ptr<float>();
    GM_ADDR output_addr = output.data_ptr<float>();
    int m = left.size(0);
    int n = right.size(1);
    int k = left.size(1);

    run_test_matmul_add<
        Arch::AtlasA2,
        layout::RowMajor,
        layout::RowMajor,
        layout::RowMajor
    >(left_addr, right_addr, output_addr, m, n, k);
}