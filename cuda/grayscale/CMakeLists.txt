cmake_minimum_required(VERSION 3.18)
project(cuda_opencv_project LANGUAGES CXX CUDA)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find OpenCV (installed via vcpkg)
find_package(OpenCV REQUIRED)

# Add executable
add_executable(grayscale main.cu)

# Link OpenCV
target_link_libraries(grayscale PRIVATE ${OpenCV_LIBS})

# Ensure NVCC compiles the CUDA code
set_target_properties(grayscale PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
