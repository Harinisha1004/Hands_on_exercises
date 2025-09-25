cmake_minimum_required(VERSION 3.18)
project(rgb_conversion_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD)
set(CMAKE_CUDA_STANDARD)

find_package(OpenCV REQUIRED)

add_executable(rgb_conversion main.cu)

target_link_libraries(rgb_conversion PRIVATE ${OpenCV_LIBS})

set_target_properties(rgb_conversion PROPERTIES CUDA_SEPARABLE_COMPILATION ON)