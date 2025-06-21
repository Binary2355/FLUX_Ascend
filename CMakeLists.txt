cmake_minimum_required(VERSION 3.19.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(flux_ascend)

set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/ascend-toolkit/latest CACHE PATH "")
# get the ${ASCEND_CANN_PACKAGE_PATH}'s parent path
get_filename_component(ASCEND_PATH ${ASCEND_CANN_PACKAGE_PATH} DIRECTORY)
set(CANN_PATHS "")

set (CANN_PATHS /usr/local/Ascend/ascend-toolkit/latest)

set(ASCEND_PYTHON_EXECUTABLE python3 CACHE STRING "")

# find the arch of the device
execute_process(
  COMMAND uname -m
  COMMAND tr -d '\n'
  OUTPUT_VARIABLE ARCH)
set(ASCEND_TENSOR_COMPILER_PATH ${ASCEND_CANN_PACKAGE_PATH}/compiler)
set(ASCEND_CCEC_COMPILER_PATH ${ASCEND_TENSOR_COMPILER_PATH}/ccec_compiler/bin)
set(ASCEND_AUTOGEN_PATH ${CMAKE_BINARY_DIR}/autogen)
set(ASCEND_FRAMEWORK_TYPE tensorflow)
file(MAKE_DIRECTORY ${ASCEND_AUTOGEN_PATH})
set(CUSTOM_COMPILE_OPTIONS "custom_compile_options.ini")
execute_process(COMMAND rm -rf ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}
                COMMAND touch ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS})

include(cmake/func.cmake)

add_library(intf_pub INTERFACE)
target_compile_options(
   intf_pub
  INTERFACE
    -fPIC                       # 生成位置无关代码（共享库必需）
    -fvisibility=hidden         # 隐藏所有符号（默认不导出）
    -fvisibility-inlines-hidden # 隐藏内联函数符号
    # 配置相关选项
    $<$<CONFIG:Release>:-O2>    # Release模式使用-O2优化
    $<$<CONFIG:Debug>:-O0 -g>   # Debug模式禁用优化(-O0)并生成调试信息(-g)
    # C++特定选项
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>  # C++代码使用C++11标准
    # Debug模式下的C++额外选项
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:
        -ftrapv     # 整数溢出时触发陷阱
        -fstack-check> # 栈溢出检测
    # C语言特定选项
    $<$<COMPILE_LANGUAGE:C>:
        -pthread        # 启用POSIX线程支持
        -Wfloat-equal   # 警告浮点数直接比较
        -Wshadow        # 警告变量遮蔽
        -Wformat=2      # 加强格式字符串检查
        -Wno-deprecated # 禁用废弃声明警告
        -Wextra>        # 额外警告
    # 栈保护（根据编译器版本选择）
    $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,
        -fstack-protector-strong,  # GCC>4.8.5用强栈保护
        -fstack-protector-all>     # 旧版GCC用全栈保护
)
target_compile_definitions(
  intf_pub INTERFACE 
    _GLIBCXX_USE_CXX11_ABI=0  # 使用旧版GLIBCXX ABI（兼容性）
    $<$<CONFIG:Release>:_FORTIFY_SOURCE=2> # Release模式开启缓冲区溢出保护
)
target_include_directories(
  intf_pub INTERFACE ${ASCEND_CANN_PACKAGE_PATH}/include
                     ${PROJECT_SOURCE_DIR}/include
                     ${PROJECT_SOURCE_DIR}/include/host
                     ${PROJECT_SOURCE_DIR}/include/kernel)
if(CANN_PATHS)
  if(${ARCH} STREQUAL "aarch64")
    target_include_directories(intf_pub
                               INTERFACE ${CANN_PATHS}/aarch64-linux/include)
    target_link_directories(intf_pub INTERFACE
                            ${CANN_PATHS}/aarch64-linux/lib64)
  else()
    target_include_directories(intf_pub
                               INTERFACE ${CANN_PATHS}/x86_64-linux/include)
    target_link_directories(intf_pub INTERFACE ${CANN_PATHS}/x86_64-linux/lib64)
  endif()
endif()

target_link_options(
   intf_pub
   INTERFACE
    $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-pie>  # 可执行文件启用PIE
    $<$<CONFIG:Release>:-s>  # Release模式移除符号表（减小体积）
    # 安全加固选项
    -Wl,-z,relro        # 重定位只读(RELRO)
    -Wl,-z,now          # 立即绑定(Now Binding)
    -Wl,-z,noexecstack  # 禁止栈执行(NX)
)

include(ExternalProject)
ExternalProject_Add(catlass_external
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/catlass
    BINARY_DIR ${CMAKE_BINARY_DIR}/catlass_build
    CMAKE_ARGS 
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/catlass_install
        -DASCEND_HOME_PATH=$ENV{ASCEND_HOME_PATH}
    INSTALL_COMMAND ""
)
add_library(catlass INTERFACE)
add_dependencies(catlass catlass_external)
target_include_directories(catlass INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/catlass/include
    ${CMAKE_CURRENT_BINARY_DIR}/catlass_install/include
    ${PROJECT_SOURCE_DIR}/third_party/catlass/examples/common
)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/ascend_ops)
add_subdirectory(${PROJECT_SOURCE_DIR}/catlass_ops)
