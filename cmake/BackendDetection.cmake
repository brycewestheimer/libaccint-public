# BackendDetection.cmake
# Functions for detecting CUDA backend availability

# Macro to detect CUDA backend
# Using a macro instead of function so that enable_language() works at the caller's scope
macro(detect_cuda_backend)
    if(LIBACCINT_USE_CUDA STREQUAL "AUTO")
        include(CheckLanguage)
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            message(STATUS "CUDA compiler detected: ${CMAKE_CUDA_COMPILER}")
            set(LIBACCINT_USE_CUDA ON)

            # Set CUDA architectures BEFORE enable_language(CUDA)
            # Default to sm_90 (Hopper/Ada Lovelace); works on Blackwell via PTX JIT
            if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
                set(CMAKE_CUDA_ARCHITECTURES "90")
            endif()

            enable_language(CUDA)
            find_package(CUDAToolkit REQUIRED)

            message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
            message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        else()
            message(STATUS "CUDA compiler not found, disabling CUDA backend")
            set(LIBACCINT_USE_CUDA OFF)
        endif()
    elseif(LIBACCINT_USE_CUDA)
        message(STATUS "CUDA backend explicitly enabled")

        # Set CUDA architectures BEFORE enable_language(CUDA)
        # Force-set if empty or undefined (CMake may auto-init to empty)
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "90")
        endif()

        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)

        message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(STATUS "CUDA backend disabled")
    endif()
endmacro()

macro(detect_mpi_backend)
    if(LIBACCINT_USE_MPI STREQUAL "AUTO")
        find_package(MPI QUIET COMPONENTS CXX)
        if(MPI_CXX_FOUND)
            message(STATUS "MPI found, enabling MPI backend")
            set(LIBACCINT_USE_MPI ON)
        else()
            message(STATUS "MPI not found, disabling MPI backend")
            set(LIBACCINT_USE_MPI OFF)
        endif()
    elseif(LIBACCINT_USE_MPI)
        message(STATUS "MPI backend explicitly enabled")
        find_package(MPI REQUIRED COMPONENTS CXX)
        set(LIBACCINT_USE_MPI ON)
        message(STATUS "MPI CXX compiler: ${MPI_CXX_COMPILER}")
        if(MPIEXEC_EXECUTABLE)
            message(STATUS "MPI launcher: ${MPIEXEC_EXECUTABLE}")
        endif()
    else()
        message(STATUS "MPI backend disabled")
    endif()
endmacro()
