# CompilerFlags.cmake
# Compiler-specific warning, optimization, and debug flags for LibAccInt

function(apply_libaccint_compile_flags target_name)
    # Common C++ standard and features
    target_compile_features(${target_name} PUBLIC cxx_std_20)
    set_target_properties(${target_name} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    # Warning flags (CXX only - CUDA uses separate flags via -Xcompiler)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:
                /W4                 # High warning level
                /WX                 # Treat warnings as errors
                /permissive-        # Standards conformance
                /Zc:__cplusplus     # Correct __cplusplus macro
                /wd4127             # Disable: conditional expression is constant
                /wd4324             # Disable: structure was padded due to alignment
            >
        )
    else()
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:
                -Wall
                -Wextra
                -Wpedantic
                -Werror
                -Wno-unused-parameter
                -Wno-unused-variable
                -Wno-sign-compare
                -Wno-missing-field-initializers
            >
        )

        # Additional GCC-specific warnings
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-Wno-maybe-uninitialized>
            )
        endif()

        # Additional Clang-specific warnings
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-command-line-argument>
            )
        endif()
    endif()

    # Optimization flags for Release builds (CXX only - CUDA has separate flags)
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        if(MSVC)
            target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:
                    /O2             # Maximum optimization
                    /Ob2            # Inline expansion
                    /Oi             # Enable intrinsic functions
                    /Ot             # Favor fast code
                    /GL             # Whole program optimization
                >
            )
        else()
            target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:
                    -O3             # Maximum optimization
                    -ffast-math     # Fast floating point math
                    -funroll-loops  # Unroll loops
                    -fno-math-errno # Don't set errno for math functions
                >
            )

            # Enable LTO if supported
            include(CheckIPOSupported)
            check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
            if(ipo_supported)
                set_target_properties(${target_name} PROPERTIES
                    INTERPROCEDURAL_OPTIMIZATION ON
                )
            endif()
        endif()

        # Add vectorization flags based on detected ISA
        if(LIBACCINT_HAS_AVX512)
            target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-mavx512f -mavx512dq>
            )
        elseif(LIBACCINT_HAS_AVX2)
            target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
            )
        endif()
    endif()

    # Debug flags (CXX only - CUDA has separate flags)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        if(MSVC)
            target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:
                    /Od             # Disable optimization
                    /Zi             # Debug information
                    /RTC1           # Runtime checks
                >
            )
        else()
            target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:
                    -O0             # No optimization
                    -g3             # Maximum debug info
                    -fno-omit-frame-pointer
                    -fno-inline
                >
            )

            # AddressSanitizer for debug builds (optional)
            if(LIBACCINT_USE_ASAN)
                target_compile_options(${target_name} PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address -fsanitize=undefined>
                )
                target_link_options(${target_name} PRIVATE
                    -fsanitize=address
                    -fsanitize=undefined
                )
            endif()
        endif()
    endif()

    # OpenMP support
    if(LIBACCINT_USE_OPENMP)
        find_package(OpenMP REQUIRED)
        target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX)
    endif()

    # CUDA-specific flags
    if(LIBACCINT_USE_CUDA)
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                --expt-relaxed-constexpr
                --extended-lambda
                -Xcompiler=-Wall
                -Xcompiler=-Wextra
            >
        )

        if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:
                    --use_fast_math
                >
            )
        endif()
    endif()

endfunction()
