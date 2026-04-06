cmake_minimum_required(VERSION 3.14)
project(eie VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GGML_CUDA "CUDA backend" OFF)
option(GGML_HIP  "ROCm/HIP backend" OFF)

# llama.cpp submodule
if(EXISTS "${CMAKE_SOURCE_DIR}/llama.cpp/CMakeLists.txt")
    add_subdirectory(llama.cpp EXCLUDE_FROM_ALL)
endif()

add_executable(eie-server
    server/main.cpp
    server/api.cpp
    backends/cpu_backend.cpp
)

target_include_directories(eie-server PRIVATE ${CMAKE_SOURCE_DIR})

if(EXISTS "${CMAKE_SOURCE_DIR}/llama.cpp/include")
    target_include_directories(eie-server PRIVATE
        llama.cpp/include llama.cpp/ggml/include)
    target_link_libraries(eie-server PRIVATE llama ggml common)
endif()

target_link_libraries(eie-server PRIVATE pthread)

if(GGML_CUDA)
    target_compile_definitions(eie-server PRIVATE GGML_USE_CUDA)
endif()
if(GGML_HIP)
    target_compile_definitions(eie-server PRIVATE GGML_USE_HIP)
endif()

install(TARGETS eie-server DESTINATION bin)
