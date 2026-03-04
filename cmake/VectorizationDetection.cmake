# VectorizationDetection.cmake
# Detect CPU vectorization capabilities (AVX2, AVX-512, NEON, SVE)

include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)
include(CMakePushCheckState)

function(detect_vectorization)
    cmake_push_check_state(RESET)

    set(LIBACCINT_HAS_AVX512 OFF PARENT_SCOPE)
    set(LIBACCINT_HAS_AVX2 OFF PARENT_SCOPE)
    set(LIBACCINT_HAS_NEON OFF PARENT_SCOPE)
    set(LIBACCINT_HAS_SVE OFF PARENT_SCOPE)
    set(LIBACCINT_VECTOR_ISA "none" PARENT_SCOPE)
    set(LIBACCINT_VECTOR_WIDTH 1 PARENT_SCOPE)

    # Check for x86_64 AVX-512
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
        set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512dq")
        check_cxx_source_runs("
            #include <immintrin.h>
            int main() {
                __m512d a = _mm512_set1_pd(1.0);
                __m512d b = _mm512_set1_pd(2.0);
                __m512d c = _mm512_add_pd(a, b);
                return 0;
            }
        " HAS_AVX512)

        if(HAS_AVX512)
            message(STATUS "AVX-512 support detected")
            set(LIBACCINT_HAS_AVX512 ON PARENT_SCOPE)
            set(LIBACCINT_VECTOR_ISA "AVX512" PARENT_SCOPE)
            set(LIBACCINT_VECTOR_WIDTH 8 PARENT_SCOPE)
            cmake_pop_check_state()
            return()
        endif()

        # Check for AVX2
        set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma")
        check_cxx_source_runs("
            #include <immintrin.h>
            int main() {
                __m256d a = _mm256_set1_pd(1.0);
                __m256d b = _mm256_set1_pd(2.0);
                __m256d c = _mm256_add_pd(a, b);
                __m256d d = _mm256_fmadd_pd(a, b, c);
                return 0;
            }
        " HAS_AVX2)

        if(HAS_AVX2)
            message(STATUS "AVX2 support detected")
            set(LIBACCINT_HAS_AVX2 ON PARENT_SCOPE)
            set(LIBACCINT_VECTOR_ISA "AVX2" PARENT_SCOPE)
            set(LIBACCINT_VECTOR_WIDTH 4 PARENT_SCOPE)
            cmake_pop_check_state()
            return()
        endif()
    endif()

    # Check for ARM NEON
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm)|(ARM)|(aarch64)|(AARCH64)")
        set(CMAKE_REQUIRED_FLAGS "")
        check_cxx_source_runs("
            #include <arm_neon.h>
            int main() {
                float64x2_t a = vdupq_n_f64(1.0);
                float64x2_t b = vdupq_n_f64(2.0);
                float64x2_t c = vaddq_f64(a, b);
                return 0;
            }
        " HAS_NEON)

        if(HAS_NEON)
            message(STATUS "ARM NEON support detected")
            set(LIBACCINT_HAS_NEON ON PARENT_SCOPE)
            set(LIBACCINT_VECTOR_ISA "NEON" PARENT_SCOPE)
            set(LIBACCINT_VECTOR_WIDTH 2 PARENT_SCOPE)
            cmake_pop_check_state()
            return()
        endif()

        # Check for ARM SVE (Scalable Vector Extension)
        set(CMAKE_REQUIRED_FLAGS "-march=armv8-a+sve")
        check_cxx_source_runs("
            #include <arm_sve.h>
            int main() {
                svfloat64_t a = svdup_f64(1.0);
                svfloat64_t b = svdup_f64(2.0);
                svbool_t pg = svptrue_b64();
                svfloat64_t c = svadd_f64_z(pg, a, b);
                return 0;
            }
        " HAS_SVE)

        if(HAS_SVE)
            message(STATUS "ARM SVE support detected")
            set(LIBACCINT_HAS_SVE ON PARENT_SCOPE)
            set(LIBACCINT_VECTOR_ISA "SVE" PARENT_SCOPE)
            set(LIBACCINT_VECTOR_WIDTH 4 PARENT_SCOPE)
            cmake_pop_check_state()
            return()
        endif()
    endif()

    message(STATUS "No vectorization support detected")
    cmake_pop_check_state()
endfunction()
