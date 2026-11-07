# The CUDA fixture of the reference scans (reference_scans.h), against a crit checkout: Core's CUDA backend, the FP64 host
# oracle (recurrent-state-recovery's recurrent_recovery.h) and the commit kernels the scans pair with
# (recurrent-state-recovery's recurrent_cuda.cu stays in crit).
#   cmake -S tests/cuda -B build/cuda-tests -DCRIT_ROOT=/path/to/crit -DCMAKE_CUDA_ARCHITECTURES=120a
#   cmake --build build/cuda-tests && ctest --test-dir build/cuda-tests --output-on-failure
cmake_minimum_required(VERSION 3.28)
project(recurrent_scans_cuda_tests LANGUAGES CXX CUDA)
include(CTest)
set(CRIT_ROOT "" CACHE PATH "a crit checkout")
if(NOT EXISTS "${CRIT_ROOT}/core/CMakeLists.txt")
  message(FATAL_ERROR "CRIT_ROOT must name a crit checkout (core/, inference/recurrent-state-recovery/code/)")
endif()
find_package(CUDAToolkit REQUIRED)
set(CRIT_CORE_CUDA ON CACHE BOOL "Build the explicit CUDA backend services" FORCE)
add_subdirectory("${CRIT_ROOT}/core" "${CMAKE_CURRENT_BINARY_DIR}/core")
set(_recovery "${CRIT_ROOT}/inference/recurrent-state-recovery/code")
add_library(reference_scans STATIC reference_scans.cu "${_recovery}/recurrent_cuda.cu")
target_include_directories(reference_scans PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${_recovery}")
target_compile_features(reference_scans PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(reference_scans PUBLIC CUDA::cudart)
add_executable(reference_scans_tests reference_scans_tests.cpp)
target_compile_features(reference_scans_tests PRIVATE cxx_std_20)
target_link_libraries(reference_scans_tests PRIVATE reference_scans crit::cuda)
foreach(_case oracle deferred rejects)
  add_test(NAME reference_scans_${_case} COMMAND reference_scans_tests ${_case})
  set_tests_properties(reference_scans_${_case} PROPERTIES TIMEOUT 120)
endforeach()
