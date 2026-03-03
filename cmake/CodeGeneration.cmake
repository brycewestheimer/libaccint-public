# CMake module for integrating LibAccInt code generation
#
# This module provides functions for regenerating kernels via the
# libaccint-codegen tool and for using pre-generated kernels.
#
# Options:
#   LIBACCINT_REGENERATE_KERNELS - If ON, regenerate kernels at build time
#   LIBACCINT_MAX_AM - Maximum angular momentum for generated kernels (default: 4)
#                      0=S, 1=P, 2=D, 3=F, 4=G
#   LIBACCINT_K_RANGES - Contraction range boundaries (default: "3,6")
#                        Defines SmallK/MediumK/LargeK thresholds
#
# Functions:
#   libaccint_add_generated_kernels(TARGET) - Add generated kernels to a target

# Early return when the codegen package is absent (alpha / stripped builds).
# Check for a key source file rather than the directory itself, since
# untracked files (e.g. __pycache__, .egg-info) may keep the directory present.
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/cli.py")
    message(STATUS "Code generation: codegen package not present (alpha build)")
    function(libaccint_add_codegen_target)
    endfunction()
    return()
endif()

option(LIBACCINT_REGENERATE_KERNELS "Regenerate kernels from codegen" OFF)

# Use the project-level LIBACCINT_MAX_AM / LIBACCINT_K_RANGES if already set,
# otherwise provide defaults here so this module works standalone.
if(NOT DEFINED LIBACCINT_MAX_AM)
    set(LIBACCINT_MAX_AM "4" CACHE STRING
        "Maximum angular momentum for generated kernels (0=S, 1=P, 2=D, 3=F, 4=G)")
    set_property(CACHE LIBACCINT_MAX_AM PROPERTY STRINGS "2" "3" "4")
endif()
if(NOT DEFINED LIBACCINT_K_RANGES)
    set(LIBACCINT_K_RANGES "3,6" CACHE STRING
        "K-range boundaries for contraction-aware dispatch (comma-separated)")
endif()

string(REPLACE "," ";" _LIBACCINT_K_RANGES_LIST "${LIBACCINT_K_RANGES}")
list(LENGTH _LIBACCINT_K_RANGES_LIST _LIBACCINT_K_RANGES_LEN)
if(NOT _LIBACCINT_K_RANGES_LEN EQUAL 2)
    message(FATAL_ERROR
        "LIBACCINT_K_RANGES must contain two comma-separated integers (got '${LIBACCINT_K_RANGES}')")
endif()
list(GET _LIBACCINT_K_RANGES_LIST 0 _LIBACCINT_SMALL_K_MAX)
list(GET _LIBACCINT_K_RANGES_LIST 1 _LIBACCINT_MEDIUM_K_MAX)
string(STRIP "${_LIBACCINT_SMALL_K_MAX}" _LIBACCINT_SMALL_K_MAX)
string(STRIP "${_LIBACCINT_MEDIUM_K_MAX}" _LIBACCINT_MEDIUM_K_MAX)
if(NOT _LIBACCINT_SMALL_K_MAX MATCHES "^[0-9]+$" OR
   NOT _LIBACCINT_MEDIUM_K_MAX MATCHES "^[0-9]+$")
    message(FATAL_ERROR
        "LIBACCINT_K_RANGES must be numeric (got '${LIBACCINT_K_RANGES}')")
endif()
if(_LIBACCINT_SMALL_K_MAX LESS 1 OR _LIBACCINT_MEDIUM_K_MAX LESS 1 OR
   _LIBACCINT_SMALL_K_MAX GREATER_EQUAL _LIBACCINT_MEDIUM_K_MAX)
    message(FATAL_ERROR
        "LIBACCINT_K_RANGES requires 1 <= small < medium (got '${LIBACCINT_K_RANGES}')")
endif()
set(_LIBACCINT_K_RANGES_NORMALIZED "${_LIBACCINT_SMALL_K_MAX},${_LIBACCINT_MEDIUM_K_MAX}")

# Find Python for code generation
find_package(Python3 COMPONENTS Interpreter)

# Directory containing generated kernels
set(LIBACCINT_GENERATED_KERNELS_DIR "${CMAKE_SOURCE_DIR}/generated" CACHE PATH
    "Directory containing pre-generated kernels")

# Output directory for regenerated kernels (during build)
set(LIBACCINT_CODEGEN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/generated_kernels" CACHE PATH
    "Output directory for regenerated kernels")

# Function to configure code generation
function(libaccint_configure_codegen)
    if(LIBACCINT_REGENERATE_KERNELS)
        if(NOT Python3_FOUND)
            message(FATAL_ERROR "Python3 is required for kernel regeneration")
        endif()
        
        # Check if libaccint-codegen is installed
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import libaccint_codegen; print(libaccint_codegen.__version__)"
            RESULT_VARIABLE CODEGEN_CHECK_RESULT
            OUTPUT_VARIABLE CODEGEN_VERSION
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(NOT CODEGEN_CHECK_RESULT EQUAL 0)
            message(STATUS "libaccint-codegen not found, installing from codegen/")
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -m pip install -e "${CMAKE_SOURCE_DIR}/codegen"
                RESULT_VARIABLE INSTALL_RESULT
            )
            if(NOT INSTALL_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to install libaccint-codegen")
            endif()
        else()
            message(STATUS "Found libaccint-codegen version ${CODEGEN_VERSION}")
        endif()
    endif()
endfunction()

# Function to add code generation as a build step
function(libaccint_add_codegen_target)
    if(NOT LIBACCINT_REGENERATE_KERNELS)
        message(STATUS "Using pre-generated kernels from ${LIBACCINT_GENERATED_KERNELS_DIR}")
        return()
    endif()
    
    message(STATUS "Configuring kernel regeneration (max-am=${LIBACCINT_MAX_AM}, k-ranges=${_LIBACCINT_K_RANGES_NORMALIZED})")
    
    # Determine which backends to generate
    set(CODEGEN_BACKENDS "cpu")
    if(LIBACCINT_USE_CUDA)
        list(APPEND CODEGEN_BACKENDS "cuda")
    endif()
    string(REPLACE ";" " " CODEGEN_BACKENDS_STR "${CODEGEN_BACKENDS}")
    
    # Create output directory
    file(MAKE_DIRECTORY ${LIBACCINT_CODEGEN_OUTPUT_DIR})
    
    # Add custom target for code generation
    add_custom_target(codegen
        COMMAND ${Python3_EXECUTABLE} -m libaccint_codegen.cli
            --max-am ${LIBACCINT_MAX_AM}
            --k-ranges ${_LIBACCINT_K_RANGES_NORMALIZED}
            --backends ${CODEGEN_BACKENDS}
            --output ${LIBACCINT_CODEGEN_OUTPUT_DIR}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Regenerating kernel code for backends: ${CODEGEN_BACKENDS_STR}"
        VERBATIM
    )
    
    # Create stamp file to track regeneration
    set(CODEGEN_STAMP_FILE "${LIBACCINT_CODEGEN_OUTPUT_DIR}/.codegen_stamp")
    
    add_custom_command(
        OUTPUT ${CODEGEN_STAMP_FILE}
        COMMAND ${Python3_EXECUTABLE} -m libaccint_codegen.cli
            --max-am ${LIBACCINT_MAX_AM}
            --k-ranges ${_LIBACCINT_K_RANGES_NORMALIZED}
            --backends ${CODEGEN_BACKENDS}
            --output ${LIBACCINT_CODEGEN_OUTPUT_DIR}
        COMMAND ${CMAKE_COMMAND} -E touch ${CODEGEN_STAMP_FILE}
        DEPENDS
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/cli.py
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/config.py
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/kernel_spec.py
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/cost_model.py
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/rys.py
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/backends/base.py
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/backends/cpu.py
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/backends/cuda.py
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/backends/hip.py
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/strategies/gpu_strategy.py
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/strategies/memory_strategy.py
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/templates/overlap_kernel.cu.j2
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/templates/kinetic_kernel.cu.j2
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/templates/nuclear_kernel.cu.j2
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/templates/eri_kernel.cu.j2
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/templates/eri_kernel_cooperative.cu.j2
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/templates/cuda/registry.cu.j2
            ${CMAKE_SOURCE_DIR}/codegen/libaccint_codegen/templates/cuda/generated_sources.cmake.j2
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Regenerating kernels..."
        VERBATIM
    )
    
    add_custom_target(codegen_auto DEPENDS ${CODEGEN_STAMP_FILE})
    
    message(STATUS "  Target 'codegen' added for manual regeneration")
    message(STATUS "  Output directory: ${LIBACCINT_CODEGEN_OUTPUT_DIR}")
endfunction()

# Function to collect generated kernel sources
function(libaccint_get_generated_sources OUT_VAR BACKEND)
    if(LIBACCINT_REGENERATE_KERNELS)
        set(KERNELS_DIR "${LIBACCINT_CODEGEN_OUTPUT_DIR}/${BACKEND}")
    else()
        set(KERNELS_DIR "${LIBACCINT_GENERATED_KERNELS_DIR}/${BACKEND}")
    endif()
    
    if(BACKEND STREQUAL "cpu")
        file(GLOB _SOURCES "${KERNELS_DIR}/*.hpp")
    elseif(BACKEND STREQUAL "cuda")
        file(GLOB _SOURCES "${KERNELS_DIR}/*.cu")
    elseif(BACKEND STREQUAL "hip")
        file(GLOB _SOURCES "${KERNELS_DIR}/*.hip.cpp")
    endif()
    
    set(${OUT_VAR} ${_SOURCES} PARENT_SCOPE)
endfunction()

# Function to add generated kernels to a target
function(libaccint_add_generated_kernels TARGET BACKEND)
    libaccint_get_generated_sources(_SOURCES ${BACKEND})
    
    if(_SOURCES)
        target_sources(${TARGET} PRIVATE ${_SOURCES})
        
        if(LIBACCINT_REGENERATE_KERNELS)
            add_dependencies(${TARGET} codegen_auto)
        endif()
        
        list(LENGTH _SOURCES _N_SOURCES)
        message(STATUS "Added ${_N_SOURCES} generated ${BACKEND} kernels to ${TARGET}")
    endif()
endfunction()

# Auto-configure if this module is included
libaccint_configure_codegen()
