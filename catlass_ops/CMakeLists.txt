file(GLOB MY_CATLASS_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
)
if(BUILD_STAGE EQUAL 0)
  set(BISHENG_COMPILER_OPTIONS
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
    -I${PROJECT_SOURCE_DIR}/include/kernel
    -I${PROJECT_SOURCE_DIR}/third_party/catlass/examples/common
    -I${PROJECT_SOURCE_DIR}/third_party/catlass/include
  )

  set(ARCH "dav-c220")
  set(CMAKE_BISHENG_COMPILER "bisheng")
  file(GLOB_RECURSE CATLASS_INCLUDE_FILES ${PROJECT_SOURCE_DIR}/third_party/catlass/include/*.hpp)
  file(GLOB_RECURSE CATLASS_EXAMPLES_COMMON_INCLUDE_FILES ${PROJECT_SOURCE_DIR}/third_party/catlass/examples/common/*.hpp)

  set(OBJECT_FILES)
  foreach(src_file ${MY_CATLASS_SRC})
    get_filename_component(src_name ${src_file} NAME_WE)
    set(obj_file "${FLUX_ASCEND_PATH}/${src_name}.o")
    list(APPEND OBJECT_FILES ${obj_file})

    add_custom_command(
      OUTPUT ${obj_file}
      COMMAND ${CMAKE_BISHENG_COMPILER} 
        --cce-aicore-arch=${ARCH}
        ${BISHENG_COMPILER_OPTIONS}
        -c ${src_file} -o ${obj_file}
      DEPENDS ${src_file} ${CATLASS_INCLUDE_FILES} ${CATLASS_EXAMPLES_COMMON_INCLUDE_FILES}
      COMMENT "Compiling ${src_file} for NPU"
    )
  endforeach()

  add_custom_command(
    OUTPUT ${FLUX_ASCEND_PATH}/libcatlass_all_ops.so
    COMMAND ${CMAKE_BISHENG_COMPILER} 
      --cce-aicore-arch=${ARCH}
      -shared
      ${OBJECT_FILES}
      -o ${FLUX_ASCEND_PATH}/libcatlass_all_ops.so
      -L$ENV{ASCEND_HOME_PATH}/lib64
      -lruntime -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase -lprofapi
    DEPENDS ${OBJECT_FILES}
    COMMENT "Linking NPU shared library"
  )

  add_custom_target(catlass_all_ops_target ALL
    DEPENDS ${FLUX_ASCEND_PATH}/libcatlass_all_ops.so
  )

  add_library(catlass_all_ops SHARED IMPORTED)
  set_target_properties(catlass_all_ops PROPERTIES
      IMPORTED_LOCATION ${FLUX_ASCEND_PATH}/libcatlass_all_ops.so
  )

  install(FILES ${FLUX_ASCEND_PATH}/libcatlass_all_ops.so
    DESTINATION lib
  )
elseif(BUILD_STAGE EQUAL 2)
  target_sources(_C PRIVATE 
    ${FLUX_ASCEND_PATH}/libcatlass_all_ops.so
  )
endif()
