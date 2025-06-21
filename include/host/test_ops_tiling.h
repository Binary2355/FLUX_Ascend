/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef TEST_OPS_TILING_H
#define TEST_OPS_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TestOpsTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, core_used);
    TILING_DATA_FIELD_DEF(uint32_t, core_data);
    TILING_DATA_FIELD_DEF(uint32_t, copy_loop);
    TILING_DATA_FIELD_DEF(uint32_t, copy_tail);
    TILING_DATA_FIELD_DEF(uint32_t, last_copy_loop);
    TILING_DATA_FIELD_DEF(uint32_t, last_copy_tail);
    TILING_DATA_FIELD_DEF(uint32_t, box_number);
    TILING_DATA_FIELD_DEF(uint32_t, available_ub_size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TestOps, TestOpsTilingData)
}
#endif // TEST_OPS_TILING_H