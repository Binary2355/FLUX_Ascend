/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "test_ops.h"
using namespace AscendC;
using namespace TestOps;

constexpr int32_t BUFFER_NUM = 2; // double buffer

__aicore__ inline void KernelTestOps::Process()
{
    uint32_t core_id = GetBlockIdx();
    if (core_id > this->core_used) {
        return;
    }
    if (core_id != (this->core_used -1)) {
        for (int32_t i = 0; i < this->copy_loop; i++) {
            uint64_t address = i * this->available_ub_size;
            Compute(i, this->available_ub_size, address);
        }
        if (this->copy_tail != 0) {
            uint64_t address = this->copy_loop * this->available_ub_size;
            Compute(this->copy_loop, this->copy_tail, address);
        }
    } else {
        for (int32_t i = 0; i < this->last_copy_loop; i++) {
            uint64_t address = i * this->available_ub_size;
            Compute(i, this->available_ub_size, address);
        }
        if (this->last_copy_tail != 0) {
            uint64_t address = this->last_copy_loop * this->available_ub_size;
            Compute(this->last_copy_loop, this->last_copy_tail, address);
        }
    }
}


__aicore__ inline void KernelTestOps::Compute(int32_t progress, int32_t tensor_size, uint64_t address)
{
    input_x = inQueueBOXES.Get<DTYPE_Y>();
    input_y = inQueuePTS.Get<DTYPE_Y>();
    zLocal = outQueueOUTPUT.Get<DTYPE_Y>();
    DataCopyParams copyParams_out{1, (uint16_t)(tensor_size * sizeof(DTYPE_X)), 0, 0};
    DataCopyParams copyParams_in{1, (uint16_t)(tensor_size* sizeof(DTYPE_X)), 0, 0};
    DataCopyParams copyParams_box{1, (uint16_t)(tensor_size * sizeof(DTYPE_X)), 0, 0};
    DataCopyPadParams padParams{true, 0, 0, 0};
    DataCopyPad(input_x, ptsGm[address], copyParams_in, padParams);
    DataCopyPad(input_y, boxesGm[address], copyParams_box, padParams);
    pipe_barrier(PIPE_ALL);
    Add(zLocal, input_x, input_y, tensor_size);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    DataCopyPad(ptsGm[address], zLocal, copyParams_out);
}

extern "C" __global__ __aicore__ void test_ops(GM_ADDR x, GM_ADDR y,
                                               GM_ADDR x_ref,
                                               GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelTestOps op(x, y, &tiling_data);
    op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void test_ops_do(uint32_t blockDim, void* l2ctrl,
                 void* stream, uint8_t* boxes, uint8_t* pts, uint8_t* boxes_idx_of_points,
                 uint8_t* workspace, uint8_t* tiling)
{
    test_ops<<<blockDim, l2ctrl, stream>>>(boxes, pts, boxes_idx_of_points, workspace, tiling);
}
#endif