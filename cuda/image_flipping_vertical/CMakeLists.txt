cmake_minimum_required(VERSION 3.18)
project(image_flipping_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD)
set(CMAKE_CUDA_STANDARD)

find_package(OpenCV REQUIRED)

add_executable(image_flipping main.cu)

target_link_libraries(image_flipping PRIVATE ${OpenCV_LIBS})

set_target_properties(image_flipping PROPERTIES CUDA_SEPARABLE_COMPILATION ON)