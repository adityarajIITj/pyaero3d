cmake_minimum_required(VERSION 3.15)
project(pyaero3d_physics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler optimization flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast)
else()
    add_compile_options(-O3 -mavx2 -mfma -ffast-math)
endif()

include_directories(include)

add_library(pyaero3d_physics SHARED
    src/pyaero3d_physics.cpp
)

set_target_properties(pyaero3d_physics PROPERTIES
    OUTPUT_NAME "pyaero3d_physics"
    PREFIX ""
)
