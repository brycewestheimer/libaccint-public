# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

# =============================================================================
# LibAccInt Python Bindings (pybind11)
# =============================================================================

cmake_minimum_required(VERSION 3.25)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG QUIET)

# If pybind11 is not found, only fetch it for developer builds.
if(NOT pybind11_FOUND)
    if(NOT LIBACCINT_ALLOW_FETCHCONTENT)
        message(FATAL_ERROR
            "pybind11 not found and LIBACCINT_ALLOW_FETCHCONTENT=OFF. "
            "Install pybind11 in the build environment.")
    endif()
    message(STATUS "pybind11 not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.12.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Get NumPy include directory
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NUMPY_NOT_FOUND
)

if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "NumPy not found. Please install numpy: pip install numpy")
endif()

message(STATUS "NumPy include directory: ${NUMPY_INCLUDE_DIR}")

# =============================================================================
# Python Extension Module: _core
# =============================================================================

pybind11_add_module(_core
    src/bindings.cpp
    src/core_types.cpp
    src/basis_bindings.cpp
    src/operator_bindings.cpp
    src/buffer_bindings.cpp
    src/engine_bindings.cpp
    src/consumer_bindings.cpp
    src/screening_bindings.cpp
    src/convenience_api.cpp
    src/advanced_bindings.cpp
)

target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${NUMPY_INCLUDE_DIR}
)

# Link against libaccint
target_link_libraries(_core PRIVATE
    libaccint
)

# Set output directory to build tree to avoid polluting the source tree
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/libaccint"
)

# Platform-specific settings
if(APPLE)
    set_target_properties(_core PROPERTIES
        INSTALL_RPATH "@loader_path"
    )
else()
    set_target_properties(_core PROPERTIES
        INSTALL_RPATH "$ORIGIN"
    )
endif()

# =============================================================================
# Installation
# =============================================================================

# Install the Python package to the Python site-packages
install(TARGETS _core
    LIBRARY DESTINATION ${Python3_SITEARCH}/libaccint
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/libaccint/
    DESTINATION ${Python3_SITEARCH}/libaccint
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE
)
