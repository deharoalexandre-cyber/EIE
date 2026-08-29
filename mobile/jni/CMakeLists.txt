# Moteur natif « EIE Mobile » — pont JNI (Soleau §3.4).
#
# On NE recompile pas llama.cpp ici : on linke les .so prébuildés (fork
# turbo-tan/llama.cpp-tq3, construits via la toolchain Snapdragon) déposés dans
# jniLibs/, et on compile uniquement notre petit native_inference.cpp.

cmake_minimum_required(VERSION 3.22.1)
project("trustid_inference" CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Chemins
set(JNILIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}")
# Sources du fork (pour les headers de la lib `common`), hors module app.
set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../native/llama.cpp-tq3")

# Libs prébuildées importées (présentes dans jniLibs/${ANDROID_ABI}).
# Les backends ggml (cpu/opencl/hexagon/htp-*) sont chargés au runtime par
# ggml_backend_load_all() — pas besoin de les linker explicitement.
foreach(LIB llama mtmd llama-common ggml ggml-base)
    add_library(${LIB} SHARED IMPORTED)
    set_target_properties(${LIB} PROPERTIES
        IMPORTED_LOCATION "${JNILIBS_DIR}/lib${LIB}.so")
endforeach()

add_library(trustid_inference SHARED native_inference.cpp)

target_include_directories(trustid_inference PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"  # headers installés (llama/ggml/mtmd…)
    "${LLAMA_DIR}/common")                 # common.h / sampling.h

find_library(log-lib log)

target_link_libraries(trustid_inference
    llama mtmd llama-common ggml ggml-base
    ${log-lib})
