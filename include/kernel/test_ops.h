#ifndef TEST_OPS_H
#define TEST_OPS_H
using namespace AscendC;

namespace TestOps {
class KernelTestOps {
public:
    __aicore__ inline KernelTestOps(GM_ADDR x, GM_ADDR y, TestOpsTilingData *tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zeronumber!");
        this->core_used = tiling_data->core_used;
        this->core_data = tiling_data->core_data;
        this->copy_loop = tiling_data->copy_loop;
        this->copy_tail = tiling_data->copy_tail;
        this->last_copy_loop = tiling_data->last_copy_loop;
        this->last_copy_tail = tiling_data->last_copy_tail;
        this->box_number = tiling_data->box_number;
        this->available_ub_size = tiling_data->available_ub_size;

        ptsGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + GetBlockIdx() * this->core_data, this->core_data);
        boxesGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + GetBlockIdx() * this->core_data, this->core_data);
        pipe.InitBuffer(inQueuePTS, this->available_ub_size * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueBOXES, this->available_ub_size * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueOUTPUT, this->available_ub_size * sizeof(DTYPE_X));
    }
    __aicore__ inline void Process();

private:
    __aicore__ inline void Compute(int32_t progress, int32_t tensor_size, uint64_t address);

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> inQueuePTS, inQueueBOXES, outQueueOUTPUT;
    GlobalTensor<DTYPE_X> boxesGm;
    GlobalTensor<DTYPE_X> ptsGm;
    GlobalTensor<DTYPE_X> outputGm;
    uint32_t core_used;
    uint32_t core_data;
    uint32_t copy_loop;
    uint32_t copy_tail;
    uint32_t last_copy_loop;
    uint32_t last_copy_tail;
    uint32_t box_number;
    uint32_t available_ub_size;
    LocalTensor<DTYPE_X> zLocal;
    LocalTensor<DTYPE_X> input_x;
    LocalTensor<DTYPE_X> input_y;
};
}

#endif // TEST_OPS_H