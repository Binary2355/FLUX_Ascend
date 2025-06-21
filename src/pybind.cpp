#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <mutex>
#include <string>
#include "ascend_ops.h"

std::string g_opApiSoPath;
std::once_flag init_flag; // Flag for one-time initialization

void init_op_api_so_path(const std::string& path)
{
    std::call_once(init_flag, [&]() { g_opApiSoPath = path; });
}

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    m.def("_init_op_api_so_path", &init_op_api_so_path);
    m.def("npu_test_ops", &npu_test_ops);
    m.def("npu_test_matmul_add", &npu_test_matmul_add);
}