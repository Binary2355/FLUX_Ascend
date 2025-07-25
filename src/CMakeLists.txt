file(GLOB ASCEND_CSRC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
if(BUILD_STAGE EQUAL 2)
  set(Python3_USE_STATIC_LIBS FALSE)
  find_package(Python3 COMPONENTS Interpreter Development)

  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import os; import torch; print(os.path.dirname(torch.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_PATH)
  execute_process(
    COMMAND
      ${Python3_EXECUTABLE} -c
      "import os; import site; print(site.getsitepackages()[0] + '/torch_npu')"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH)
  message("TORCH_PATH is ${TORCH_PATH}")
  message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")
  message("ASCEND_CSRC_SRC is ${ASCEND_CSRC_SRC}")

  set(EXT_CXX_FLAGS "${EXT_CXX_FLAGS}")
  separate_arguments(EXT_CXX_FLAGS)
  add_library(_C SHARED ${ASCEND_CSRC_SRC})
  set_target_properties(
    _C
    PROPERTIES OUTPUT_NAME "${FLUX_ASCEND_PATH}/_C.${Python3_SOABI}"
               PREFIX ""
               SUFFIX ".so")

  if(${COMPILE_WITH_XLA})
    target_compile_definitions(_C PRIVATE COMPILE_WITH_XLA)
  endif()
  set(CANN_INCLUDE_PATH "")
  set(CANN_LIB_PATH "")
  if(CANN_PATHS)
    if(${ARCH} STREQUAL "aarch64")
      set(CANN_INCLUDE_PATH ${CANN_PATHS}/aarch64-linux/include)
      set(CANN_LIB_PATH ${CANN_PATHS}/aarch64-linux/lib64)
    else()
      set(CANN_INCLUDE_PATH ${CANN_PATHS}/x86_64-linux/include)
      set(CANN_LIB_PATH ${CANN_PATHS}/x86_64-linux/lib64)
    endif()
  endif()
  message(STATUS "CANN include path: ${CANN_INCLUDE_PATH}")
  message(STATUS "CANN lib path: ${CANN_LIB_PATH}")
  target_include_directories(
    _C
    PRIVATE ${Python3_INCLUDE_DIRS}
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/include/host
            ${PROJECT_SOURCE_DIR}/include/kernel
            ${TORCH_NPU_PATH}/include
            ${TORCH_NPU_PATH}/include/third_party/hccl/inc
            ${TORCH_NPU_PATH}/include/third_party/acl/inc
            ${TORCH_PATH}/include
            ${TORCH_PATH}/include/torch/csrc/api/include
            ${CANN_INCLUDE_PATH}
  )
  target_compile_options(
    _C
    PRIVATE -fprofile-arcs
            -ftest-coverage
            -fPIC
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Debug>:-O0
            -g>
            -fstack-protector-all
            -DTORCH_API_INCLUDE_EXTENSION_H
            -DTORCH_EXTENSION_NAME=_C
            -D_GLIBCXX_USE_CXX11_ABI=${ABI}
            -D__FILENAME__=__FILE__
            ${EXT_CXX_FLAGS})

  target_link_directories(_C PRIVATE ${TORCH_PATH}/lib ${TORCH_NPU_PATH}/lib ${CANN_LIB_PATH})
  target_link_libraries(_C PRIVATE gcov c10 torch torch_python torch_npu)
  target_link_options(
    _C
    PRIVATE
    $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-pie>
    $<$<CONFIG:Release>:-s>
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack)
endif()
