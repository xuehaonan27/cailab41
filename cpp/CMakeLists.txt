cmake_minimum_required(VERSION 3.10)
project(simd-lab)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512dq -mavx512bw -mavx512vl")

add_library(plain plain.cc)
add_library(mmx mmx.cc)
add_library(sse2 sse2.cc)
add_library(avx avx.cc)

add_executable(test main.cc)
target_link_libraries(test PUBLIC plain mmx sse2 avx)