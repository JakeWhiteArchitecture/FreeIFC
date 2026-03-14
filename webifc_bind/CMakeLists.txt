cmake_minimum_required(VERSION 3.18)
project(webifc_bind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── pybind11 ────────────────────────────────────────────────────────────────
find_package(pybind11 REQUIRED)

# ── web-ifc (static library + all FetchContent deps) ───────────────────────
# This triggers web-ifc's own CMakeLists which downloads dependencies and
# builds web-ifc-library as a static lib (the NOT EMSCRIPTEN branch).
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../web-ifc-src/src/cpp
    ${CMAKE_CURRENT_BINARY_DIR}/web-ifc-build
)

# Add REAL_T_IS_DOUBLE to web-ifc-library to match the WASM build config.
# This does NOT modify any source files — it only adds a compile definition
# to the already-defined target.
target_compile_definitions(web-ifc-library PUBLIC REAL_T_IS_DOUBLE)

# ── pybind11 extension module ──────────────────────────────────────────────
pybind11_add_module(webifc webifc_bind.cpp)
target_link_libraries(webifc PRIVATE web-ifc-library)
target_compile_definitions(webifc PRIVATE REAL_T_IS_DOUBLE)

# Suppress noisy warnings from web-ifc internals
if(NOT MSVC)
    target_compile_options(webifc PRIVATE -Wno-unused-parameter)
endif()

# ── Post-build: copy .so to project root ───────────────────────────────────
add_custom_command(TARGET webifc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:webifc>
        ${CMAKE_CURRENT_SOURCE_DIR}/../$<TARGET_FILE_NAME:webifc>
    COMMENT "Copied webifc module to project root"
)
