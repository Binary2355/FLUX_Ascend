#include <iostream>
#include <vector>

#include "helper.hpp"
#include "golden.hpp"
#include "fp16_t.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/matmul_epilogue.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp"

using namespace Catlass;
using fp16_t = op::fp16_t;

template<
    class ArchTag,
    class LayoutA,
    class LayoutB,
    class LayoutC
>
void run_test_matmul_add(GM_ADDR left, GM_ADDR right, GM_ADDR output,int m, int n, int k)
{
    // Define the layout of each matrix
    LayoutA layoutA{static_cast<typename LayoutA::Index>(m), static_cast<typename LayoutA::Index>(k)};
    LayoutB layoutB{static_cast<typename LayoutB::Index>(k), static_cast<typename LayoutB::Index>(n)};
    LayoutC layoutD{static_cast<typename LayoutC::Index>(m), static_cast<typename LayoutC::Index>(n)};

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // // Block level, define BlockMmad
    // constexpr bool enableUnitFlag = true;
    // using MmadDispatchPolicy = Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    // using L1TileShape = GemmShape<128, 256, 256>;
    // using L0TileShape = GemmShape<128, 256, 64>;
    // using AType = Gemm::GemmType<half, LayoutA>;
    // using BType = Gemm::GemmType<half, LayoutB>;
    // using CType = Gemm::GemmType<half, LayoutC>;
    // using BlockMmad = Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    // // Block level, define BlockEpilogue
    // using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2ElemWiseOneSource;
    // using XType = CType;
    // using DType = CType;
    // using ComputeType = CType;
    // constexpr uint32_t computeLength = 16384;
    // using TileElemWiseEpilogue = Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    // using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, XType, DType>;
    // using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType, XType, DType,
    //     TileElemWiseEpilogue, EpilogueTileCopy>;
    // if (m > n) {
    //     // Define BlockScheduler
    //     // Swizzle offset is 3 and direction is 0.
    //     using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    //     // Kernel level
    //     using MatmulKernel = Gemm::Kernel::MatmulEpilogue<BlockMmad, BlockEpilogue, BlockScheduler>;
    //     // Prepare params
    //     typename MatmulKernel::Arguments arguments{
    //         options.problemShape, sizeof(half), deviceA, deviceB, deviceD};
    //     using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
    //     MatmulAdapter matmul_op;
    //     size_t sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
    //     uint8_t *deviceWorkspace{nullptr};
    //     if (sizeWorkspace > 0) {
    //         ACL_CHECK(
    //             aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace,ACL_MEM_MALLOC_HUGE_FIRST));
    //     }
    //     matmul_op.Initialize(arguments, deviceWorkspace);
    //     matmul_op(stream, aicCoreNum, fftsAddr);
    //     ACL_CHECK(aclrtSynchronizeStream(stream));
    //     if (sizeWorkspace > 0) {
    //         ACL_CHECK(aclrtFree(deviceWorkspace));
    //     }

    //     // Copy the result from device to host
    //     ACL_CHECK(aclrtMemcpy(D, sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));
    // } else {
    //     // Define BlockScheduler
    //     // Swizzle offset is 3 and direction is 1.
    //     using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
    //     // Kernel level
    //     using MatmulKernel = Gemm::Kernel::MatmulEpilogue<BlockMmad, BlockEpilogue, BlockScheduler>;
    //     // Prepare params
    //     typename MatmulKernel::Arguments arguments{
    //         options.problemShape, sizeof(half), deviceA, deviceB, deviceD};
    //     using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
    //     MatmulAdapter matmul_op;
    //     size_t sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
    //     uint8_t *deviceWorkspace{nullptr};
    //     if (sizeWorkspace > 0) {
    //         ACL_CHECK(
    //             aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace,ACL_MEM_MALLOC_HUGE_FIRST));
    //     }
    //     matmul_op.Initialize(arguments, deviceWorkspace);
    //     matmul_op(stream, aicCoreNum, fftsAddr);
    //     ACL_CHECK(aclrtSynchronizeStream(stream));
    //     if (sizeWorkspace > 0) {
    //         ACL_CHECK(aclrtFree(deviceWorkspace));
    //     }

    //     // Copy the result from device to host
    //     ACL_CHECK(aclrtMemcpy(D.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));
    // }
}
