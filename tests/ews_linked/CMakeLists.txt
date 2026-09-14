cmake_minimum_required(VERSION 3.18)
project(eie_linked_forward LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
# Build only EIE's probe against an already built, matching runtime. Never
# reconfigure, overwrite or install into the supplied runtime tree.
set(EIE_SOURCE "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH "EIE source")
set(EIE_RUNTIME_SOURCE "" CACHE PATH "Matching llama.cpp headers")
set(EIE_RUNTIME_BUILD "" CACHE PATH "Existing shared runtime build")
add_executable(ews-forward "${EIE_SOURCE}/tests/ews_forward.cpp" "${EIE_SOURCE}/backends/expert_stream.cpp")
target_include_directories(ews-forward PRIVATE "${EIE_SOURCE}" "${EIE_RUNTIME_SOURCE}/include"
    "${EIE_RUNTIME_SOURCE}/src" "${EIE_RUNTIME_SOURCE}/ggml/include" "${EIE_RUNTIME_SOURCE}/vendor")
foreach(name llama ggml ggml-base ggml-cpu)
    find_library(EIE_${name} NAMES ${name} PATHS "${EIE_RUNTIME_BUILD}/src/Release"
        "${EIE_RUNTIME_BUILD}/ggml/src/Release" NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(ews-forward PRIVATE "${EIE_${name}}")
endforeach()

add_executable(routing-backend "${EIE_SOURCE}/tests/serving/backend_contract.cpp" "${EIE_SOURCE}/backends/expert_stream.cpp")
target_compile_definitions(routing-backend PRIVATE EIE_HAS_LLAMA=1)
target_include_directories(routing-backend PRIVATE "${EIE_SOURCE}" "${EIE_RUNTIME_SOURCE}/include"
    "${EIE_RUNTIME_SOURCE}/src" "${EIE_RUNTIME_SOURCE}/ggml/include" "${EIE_RUNTIME_SOURCE}/vendor" "${EIE_RUNTIME_SOURCE}/common")
find_library(EIE_COMMON NAMES llama-common PATHS "${EIE_RUNTIME_BUILD}/common/Release" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(routing-backend PRIVATE "${EIE_llama}" "${EIE_ggml}" "${EIE_ggml-base}" "${EIE_ggml-cpu}" "${EIE_COMMON}")
