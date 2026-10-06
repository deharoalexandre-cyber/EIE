cmake_minimum_required(VERSION 3.14)
project(eie VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GGML_CUDA "CUDA backend" OFF)
option(GGML_HIP  "ROCm/HIP backend" OFF)

set(EIE_LLAMA_RUNTIME "turboquant" CACHE STRING "llama.cpp runtime: turboquant or prism")
set_property(CACHE EIE_LLAMA_RUNTIME PROPERTY STRINGS turboquant prism)
if(EIE_LLAMA_RUNTIME STREQUAL "turboquant")
    set(EIE_LLAMA_SOURCE_DIR "${CMAKE_SOURCE_DIR}/llama.cpp")
elseif(EIE_LLAMA_RUNTIME STREQUAL "prism")
    set(EIE_LLAMA_SOURCE_DIR "${CMAKE_SOURCE_DIR}/llama-prism")
    if(NOT EXISTS "${EIE_LLAMA_SOURCE_DIR}/src")
        message(FATAL_ERROR "PrismML runtime missing: git submodule update --init llama-prism")
    endif()
else()
    message(FATAL_ERROR "EIE_LLAMA_RUNTIME must be turboquant or prism")
endif()

# The default submodule retains the existing placeholder build when absent.
if(EXISTS "${EIE_LLAMA_SOURCE_DIR}/src")
    # Metal is only reliable on Apple Silicon; Intel iGPUs produce wrong results.
    # (Cross-compiling for arm64 from an Intel Mac keeps Metal: see CMAKE_OSX_ARCHITECTURES.)
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
       AND NOT CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
        set(GGML_METAL OFF CACHE BOOL "" FORCE)
    endif()
    set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory("${EIE_LLAMA_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/eie-llama" EXCLUDE_FROM_ALL)
    set(EIE_HAS_LLAMA ON)
endif()

add_executable(eie-server
    server/main.cpp
    server/api.cpp
    backends/cpu_backend.cpp
)

target_include_directories(eie-server PRIVATE ${CMAKE_SOURCE_DIR})

if(EIE_HAS_LLAMA)
    if(EIE_LLAMA_RUNTIME STREQUAL "prism")
        target_sources(eie-server PRIVATE backends/expert_stream_stub.cpp)
        target_compile_definitions(eie-server PRIVATE EIE_PRISMML=1)
    else()
        target_sources(eie-server PRIVATE backends/expert_stream.cpp)
        target_compile_definitions(eie-server PRIVATE EIE_HAS_EWS=1)
    endif()
    target_include_directories(eie-server PRIVATE
        "${EIE_LLAMA_SOURCE_DIR}/include" "${EIE_LLAMA_SOURCE_DIR}/src"
        "${EIE_LLAMA_SOURCE_DIR}/ggml/include" "${EIE_LLAMA_SOURCE_DIR}/common"
        "${EIE_LLAMA_SOURCE_DIR}/vendor" "${EIE_LLAMA_SOURCE_DIR}/vendor/cpp-httplib")
    target_link_libraries(eie-server PRIVATE llama-common llama ggml cpp-httplib)
    target_compile_definitions(eie-server PRIVATE EIE_HAS_LLAMA=1 HAS_HTTPLIB=1)
endif()

option(EIE_BUILD_SERVING_TESTS "Build offline and explicit real-model serving gates" OFF)
if(EIE_BUILD_SERVING_TESTS)
    enable_testing()
    add_subdirectory(tests/serving)
    if(EIE_HAS_LLAMA)
        if(EIE_LLAMA_RUNTIME STREQUAL "prism")
            set(EIE_STREAM_SOURCE backends/expert_stream_stub.cpp)
        else()
            set(EIE_STREAM_SOURCE backends/expert_stream.cpp)
        endif()
        add_executable(serving-model-contract tests/serving/backend_contract.cpp ${EIE_STREAM_SOURCE})
        target_include_directories(serving-model-contract PRIVATE ${CMAKE_SOURCE_DIR}
            "${EIE_LLAMA_SOURCE_DIR}/include" "${EIE_LLAMA_SOURCE_DIR}/src"
            "${EIE_LLAMA_SOURCE_DIR}/ggml/include" "${EIE_LLAMA_SOURCE_DIR}/common"
            "${EIE_LLAMA_SOURCE_DIR}/vendor" "${EIE_LLAMA_SOURCE_DIR}/vendor/cpp-httplib")
        target_compile_definitions(serving-model-contract PRIVATE EIE_HAS_LLAMA=1)
        if(EIE_LLAMA_RUNTIME STREQUAL "prism")
            target_compile_definitions(serving-model-contract PRIVATE EIE_PRISMML=1)
        else()
            target_compile_definitions(serving-model-contract PRIVATE EIE_HAS_EWS=1)
        endif()
        target_link_libraries(serving-model-contract PRIVATE llama-common llama ggml cpp-httplib)
    endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(eie-server PRIVATE Threads::Threads)

if(GGML_CUDA)
    target_compile_definitions(eie-server PRIVATE GGML_USE_CUDA)
endif()
if(GGML_HIP)
    target_compile_definitions(eie-server PRIVATE GGML_USE_HIP)
endif()

install(TARGETS eie-server DESTINATION bin)

option(EIE_BUILD_EWS_TESTS "Build the matched EWS forward test" OFF)
if(EIE_BUILD_EWS_TESTS AND EIE_LLAMA_RUNTIME STREQUAL "prism")
    message(FATAL_ERROR "EWS tests require the turboquant runtime; PrismML has no EIE EWS port")
endif()
if(EIE_BUILD_EWS_TESTS AND EIE_HAS_LLAMA)
    add_executable(ews-stream-unit tests/ews_stream_unit.cpp)
    target_include_directories(ews-stream-unit PRIVATE ${CMAKE_SOURCE_DIR}
        "${EIE_LLAMA_SOURCE_DIR}/include" "${EIE_LLAMA_SOURCE_DIR}/src"
        "${EIE_LLAMA_SOURCE_DIR}/ggml/include")
    target_link_libraries(ews-stream-unit PRIVATE llama ggml)
    add_executable(ews-forward tests/ews_forward.cpp backends/expert_stream.cpp)
    target_include_directories(ews-forward PRIVATE ${CMAKE_SOURCE_DIR}
        "${EIE_LLAMA_SOURCE_DIR}/include" "${EIE_LLAMA_SOURCE_DIR}/src"
        "${EIE_LLAMA_SOURCE_DIR}/ggml/include" "${EIE_LLAMA_SOURCE_DIR}/vendor")
    target_link_libraries(ews-forward PRIVATE llama ggml)
endif()
