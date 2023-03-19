cmake_minimum_required(VERSION 3.2)
project(DNNDIFF LANGUAGES CXX CUDA)

if (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include (${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

#include directories
include_directories(${CMAKE_INCLUDE_PATH})
include_directories("include")

#initial variables
set(DNNDIF_LIBS "")
set(DNNDIF_LINK_LIBS ${CMAKE_DL_LIBS})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE DNNDIF_SRCS
  src/core/*.cc
  )
list(APPEND DNNDIFF_SRCS ${PROTO_CPPS})

file(GLOB_RECURSE DNNDIFF_CUDA_SRCS
  src/cudnn/*.cu
  )

#Compilation options
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" SUPPORT_CXX11)
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message("Build in Debug mode")
  set(CMAKE_CUDA_FLAGS "-O0 -g -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CUDA_FLAGS "-O2 -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
endif()

#set CUDA
if (NOT "${USE_CUDA}" STREQUAL "OFF")
  include(cmake/FindCUDA.cmake)
  find_cuda(${USE_CUDA})
  if (CUDA_FOUND)
    list(APPEND DNNDIFF_SRCS ${DNNDIFF_CUDA_SRCS})
    include_directories(${CUDA_INCLUDE_DIRS})
    message(STATUS "CUDA_INCLUDE_DIR=" ${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDNN)
    list(APPEND DNNDIFF_LINK_LIBS ${CUDA_CUDART_LIBRARY})
    list(APPEND DNNDIFF_LINK_LIBS ${CUDA_CUDA_LIBRARY})
    list(APPEND DNNDIFF_LINK_LIBS ${CUDA_CUDNN_LIBRARY})
    list(APPEND DNNDIFF_LINK_LIBS ${CUDA_CUBLAS_LIBRARY})
    list(APPEND DNNDIFF_LINK_LIBS ${CUDA_CUSOLVER_LIBRARY})
  else()
    message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
  endif(CUDA_FOUND)
endif()

add_library(dnndiff_runtime SHARED ${DNNDIFF_SRCS})

set_target_properties(dnndiff_runtime
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(dnndiff_runtime PUBLIC cxx_std_11)

target_link_libraries(dnndiff_runtime ${DNNDIFF_LINK_LIBS})

target_include_directories(dnndiff_runtime
  PUBLIC ${PROJECT_SOURCE_DIR}/include)

set_target_properties(dnndiff_runtime
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS dnndiff_runtime
    LIBRARY DESTINATION lib)

install (DIRECTORY ${PROJECT_SOURCE_DIR}/include
    DESTINATION .)

