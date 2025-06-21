file(GLOB MY_CATLASS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
if(BUILD_STAGE EQUAL 0)
  file(MAKE_DIRECTORY ${FLUX_ASCEND_PATH})
  file(GLOB MY_CATLASS_SRC
      ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
  )
  execute_process(
    COMMAND python3 -c "import torch; print(torch.__path__[0])"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_PATH
  )
  execute_process(
    COMMAND python3 -c "import torch_npu; print(torch_npu.__path__[0])"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
  )
  set(TORCH_INCLUDE "${TORCH_PATH}/include")
  set(TORCH_NPU_INCLUDE "${TORCH_NPU_PATH}/include")
  if(NOT EXISTS "${TORCH_INCLUDE}/ATen/ATen.h")
    message(FATAL_ERROR "ATen/ATen.h not found at: ${TORCH_INCLUDE}/ATen/ATen.h")
  endif()

  add_library(catlass_all_ops SHARED ${MY_CATLASS_SRC})
  set_target_properties(catlass_all_ops PROPERTIES
      LINK_FLAGS "-shared"
      OUTPUT_NAME "catlass_all_ops"
      VERSION 1.0.0
      SOVERSION 1
  )
  target_compile_options(catlass_all_ops PRIVATE
      -g -fPIC -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0
      -Wl,--no-undefined -Wl,--as-needed
  )
  target_include_directories(catlass_all_ops PRIVATE
      ${CANN_INCLUDE_PATH}
      ${Python3_INCLUDE_DIRS}
      $ENV{ASCEND_HOME_PATH}/compiler/tikcpp
      $ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
      $ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
      $ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
      $ENV{ASCEND_HOME_PATH}/include
      $ENV{ASCEND_HOME_PATH}/include/experiment/runtime
      $ENV{ASCEND_HOME_PATH}/include/experiment/msprof
      ${PROJECT_SOURCE_DIR}/include
      ${PROJECT_SOURCE_DIR}/third_party/catlass/examples/common
      ${PROJECT_SOURCE_DIR}/third_party/catlass/include
      ${TORCH_NPU_INCLUDE}
      ${TORCH_NPU_INCLUDE}/third_party/hccl/inc
      ${TORCH_NPU_INCLUDE}/third_party/acl/inc
      ${TORCH_INCLUDE}
      ${TORCH_INCLUDE}/torch/csrc/api/include
  )
  target_link_libraries(catlass_all_ops PRIVATE
      intf_pub exe_graph register tiling_api ascendcl catlass
  )
  set(BISHENG_COMPILER_OPTIONS
      -c  # 关键：只编译不链接
      -O2
      -std=c++17
      -xcce
      -mllvm -cce-aicore-stack-size=0x8000
      -mllvm -cce-aicore-function-stack-size=0x8000
      -mllvm -cce-aicore-record-overflow=true
      -mllvm -cce-aicore-addr-transform
      -mllvm -cce-aicore-dcci-insert-for-scalar=false
      -DL2_CACHE_HINT
      -Wno-macro-redefined
      -Wno-ignored-attributes
  )

  list(APPEND BISHENG_COMPILER_OPTIONS
    -I${Python3_INCLUDE_DIRS}
    -I$ENV{ASCEND_HOME_PATH}/compiler/tikcpp
    -I$ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    -I$ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    -I$ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    -I$ENV{ASCEND_HOME_PATH}/include
    -I$ENV{ASCEND_HOME_PATH}/include/experiment/runtime
    -I$ENV{ASCEND_HOME_PATH}/include/experiment/msprof
    -I${PROJECT_SOURCE_DIR}/include
    -I${PROJECT_SOURCE_DIR}/third_party/catlass/examples/common
    -I${PROJECT_SOURCE_DIR}/third_party/catlass/include
    -I${TORCH_NPU_INCLUDE}
    -I${TORCH_NPU_INCLUDE}/third_party/hccl/inc
    -I${TORCH_NPU_INCLUDE}/third_party/acl/inc
    -I${TORCH_INCLUDE}
    -I${TORCH_INCLUDE}/torch/csrc/api/include
  )

  foreach(src_file ${MY_CATLASS_SRC})
      get_filename_component(src_name ${src_file} NAME_WE)
      set(output_file ${FLUX_ASCEND_PATH}/${src_name}.o)
      set(ARCH "dav-c220")
      add_custom_command(
          OUTPUT ${output_file}
          COMMAND ${CMAKE_COMMAND} -E echo "Compiling ${src_file}..."
          COMMAND bisheng -v --cce-aicore-arch=${ARCH} ${BISHENG_COMPILER_OPTIONS} ${src_file} -o ${output_file}
          DEPENDS ${src_file} ${PROJECT_SOURCE_DIR}/third_party/catlass/include ${PROJECT_SOURCE_DIR}/third_party/catlass/examples/common
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          COMMENT "Compiling NPU kernel: ${src_name}"
      )
      list(APPEND KERNEL_OBJECTS ${output_file})
  endforeach()

  # 创建自定义目标来驱动所有编译任务
  add_custom_target(compile_all_kernels ALL
      DEPENDS ${KERNEL_OBJECTS}
      COMMENT "Building all NPU kernel objects"
  )

  # 将生成的对象文件添加到动态库
  target_sources(catlass_all_ops PRIVATE ${KERNEL_OBJECTS})

  # 确保动态库依赖于内核编译目标
  add_dependencies(catlass_all_ops compile_all_kernels)

  # 构建后验证
  add_custom_command(TARGET catlass_all_ops POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E echo "Verifying dynamic library..."
      COMMAND test -f $<TARGET_FILE:catlass_all_ops> || (echo "ERROR: Library not generated!"; exit 1)
      COMMAND ${CMAKE_COMMAND} -E echo "Dynamic library built successfully: $<TARGET_FILE:catlass_all_ops>"
      COMMENT "Final verification"
  )

  # 安装规则
  install(TARGETS catlass_all_ops
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
  )
elseif(BUILD_STAGE EQUAL 2)
  target_sources(_C PRIVATE 
    ${FLUX_ASCEND_PATH}/catlass_all_ops.o
  )
endif()
