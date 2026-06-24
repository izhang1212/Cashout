cmake_minimum_required(VERSION 3.16)
project(ParlayFollowerCpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native -ffast-math)
    endif()
endif()

include_directories(include)

add_executable(parlay_tests tests/test_main.cpp)
add_executable(parlay_bench bench/bench_main.cpp)

# Future: pybind11 bindings for strategy callbacks from Python
# find_package(pybind11 REQUIRED)
# pybind11_add_module(parlay_backtester bindings/py_backtester.cpp)
# target_include_directories(parlay_backtester PRIVATE include)
