# tests/CMakeLists.txt
# Test suite for LibAccInt

# Include GoogleTest
include(GoogleTest)

# =============================================================================
# Unit Tests
# =============================================================================

# Core types tests
add_executable(test_types unit/core/test_types.cpp)
target_link_libraries(test_types PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_types)

# Core backend tests
add_executable(test_backend unit/core/test_backend.cpp)
target_link_libraries(test_backend PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_backend)

# Core precision tests
add_executable(test_precision unit/core/test_precision.cpp)
target_link_libraries(test_precision PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_precision)

# Error handling tests
add_executable(test_error_handling unit/utils/test_error_handling.cpp)
target_link_libraries(test_error_handling PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_error_handling)

# Physical constants tests
add_executable(test_constants unit/utils/test_constants.cpp)
target_link_libraries(test_constants PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_constants)

# Gaussian product tests
add_executable(test_gaussian_product unit/math/test_gaussian_product.cpp)
target_link_libraries(test_gaussian_product PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_gaussian_product)

# Cartesian indices tests
add_executable(test_cartesian_indices unit/math/test_cartesian_indices.cpp)
target_link_libraries(test_cartesian_indices PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_cartesian_indices)

# Boys function tests
add_executable(test_boys_function unit/math/test_boys_function.cpp)
target_link_libraries(test_boys_function PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_boys_function)

# Float Boys function tests
add_executable(test_boys_function_float unit/math/test_boys_function_float.cpp)
target_link_libraries(test_boys_function_float PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_boys_function_float)

# Normalization tests
add_executable(test_normalization unit/math/test_normalization.cpp)
target_link_libraries(test_normalization PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_normalization)

# Rys quadrature tests
add_executable(test_rys_quadrature unit/math/test_rys_quadrature.cpp)
target_link_libraries(test_rys_quadrature PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_rys_quadrature)

# Higher angular momentum tests (f, g, h functions)
add_executable(test_higher_am unit/math/test_higher_am.cpp)
target_link_libraries(test_higher_am PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_higher_am)

# Spherical transform tests
add_executable(test_spherical_transform unit/math/test_spherical_transform.cpp)
target_link_libraries(test_spherical_transform PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_spherical_transform)

# Shell tests
add_executable(test_shell unit/basis/test_shell.cpp)
target_link_libraries(test_shell PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_shell)

# Operator types tests
add_executable(test_operator_types unit/operators/test_operator_types.cpp)
target_link_libraries(test_operator_types PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_operator_types)

# One-electron buffer tests
add_executable(test_one_electron_buffer unit/buffers/test_one_electron_buffer.cpp)
target_link_libraries(test_one_electron_buffer PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_one_electron_buffer)

# Two-electron buffer tests
add_executable(test_two_electron_buffer unit/buffers/test_two_electron_buffer.cpp)
target_link_libraries(test_two_electron_buffer PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_two_electron_buffer)

# Buffer float-precision tests
add_executable(test_buffer_float unit/buffers/test_buffer_float.cpp)
target_link_libraries(test_buffer_float PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_buffer_float)

# Buffer precision conversion tests
add_executable(test_buffer_precision unit/buffers/test_buffer_precision.cpp)
target_link_libraries(test_buffer_precision PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_buffer_precision)

# Buffer bounds-checking tests
add_executable(test_buffer_bounds unit/buffers/test_buffer_bounds.cpp)
target_link_libraries(test_buffer_bounds PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_buffer_bounds)

# Three-center buffer tests
add_executable(test_three_center_buffer unit/buffers/test_three_center_buffer.cpp)
target_link_libraries(test_three_center_buffer PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_three_center_buffer)

# ShellSet tests
add_executable(test_shell_set unit/basis/test_shell_set.cpp)
target_link_libraries(test_shell_set PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_shell_set)

# Operator class tests
add_executable(test_operator unit/operators/test_operator.cpp)
target_link_libraries(test_operator PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_operator)

# ShellSetPair tests
add_executable(test_shell_set_pair unit/basis/test_shell_set_pair.cpp)
target_link_libraries(test_shell_set_pair PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_shell_set_pair)

# OneElectronOperator tests
add_executable(test_one_electron_operator unit/operators/test_one_electron_operator.cpp)
target_link_libraries(test_one_electron_operator PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_one_electron_operator)

# ShellSetQuartet tests
add_executable(test_shell_set_quartet unit/basis/test_shell_set_quartet.cpp)
target_link_libraries(test_shell_set_quartet PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_shell_set_quartet)

# BasisSet tests
add_executable(test_basis_set unit/basis/test_basis_set.cpp)
target_link_libraries(test_basis_set PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_basis_set)

# =============================================================================
# Phase 4: Basis & Data Module Remediation Tests
# =============================================================================

# AuxiliaryBasisSet tests
add_executable(test_auxiliary_basis_set unit/basis/test_auxiliary_basis_set.cpp)
target_link_libraries(test_auxiliary_basis_set PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_auxiliary_basis_set)

# ShellSetTriple tests (Task 4.3.1)
add_executable(test_shell_set_triple unit/basis/test_shell_set_triple.cpp)
target_link_libraries(test_shell_set_triple PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_shell_set_triple)

# ShellSetQuartet utils tests (Task 4.3.2)
add_executable(test_shell_set_quartet_utils unit/basis/test_shell_set_quartet_utils.cpp)
target_link_libraries(test_shell_set_quartet_utils PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_shell_set_quartet_utils)

# Load auxiliary basis tests (Task 4.3.3)
add_executable(test_load_auxiliary_basis unit/basis/test_load_auxiliary_basis.cpp)
target_link_libraries(test_load_auxiliary_basis PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_load_auxiliary_basis)

# Element symbol tests (Task 4.3.4)
add_executable(test_element_symbols unit/data/test_element_symbols.cpp)
target_link_libraries(test_element_symbols PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_element_symbols)

# Basis normalization tests (Task 4.3.5)
add_executable(test_basis_normalization unit/basis/test_basis_normalization.cpp)
target_link_libraries(test_basis_normalization PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_basis_normalization)

# ShellSetPair batch tests (Task 4.3.6)
add_executable(test_shell_set_pair_batch unit/basis/test_shell_set_pair_batch.cpp)
target_link_libraries(test_shell_set_pair_batch PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_shell_set_pair_batch)

# Data integration tests (Task 4.3.7)
add_executable(test_data_integration unit/data/test_data_integration.cpp)
target_link_libraries(test_data_integration PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_data_integration)

# Name resolution tests
add_executable(test_name_resolution unit/data/test_name_resolution.cpp)
target_link_libraries(test_name_resolution PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_name_resolution)

# Parser error tests (Task 4.3.8)
add_executable(test_parser_errors unit/data/test_parser_errors.cpp)
target_link_libraries(test_parser_errors PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_parser_errors)

# Overlap kernel tests
add_executable(test_overlap_kernel unit/kernels/test_overlap_kernel.cpp)
target_link_libraries(test_overlap_kernel PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_overlap_kernel)

# Kinetic kernel tests
add_executable(test_kinetic_kernel unit/kernels/test_kinetic_kernel.cpp)
target_link_libraries(test_kinetic_kernel PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_kinetic_kernel)

# Nuclear kernel tests
add_executable(test_nuclear_kernel unit/kernels/test_nuclear_kernel.cpp)
target_link_libraries(test_nuclear_kernel PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_nuclear_kernel)

# ERI kernel tests (Phase 7)
add_executable(test_eri_kernel unit/kernels/test_eri_kernel.cpp)
target_link_libraries(test_eri_kernel PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_eri_kernel)

# Three-center ERI kernel tests
add_executable(test_three_center_eri unit/kernels/test_three_center_eri.cpp)
target_link_libraries(test_three_center_eri PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_three_center_eri)

# Two-center Coulomb kernel tests
add_executable(test_two_center_coulomb unit/kernels/test_two_center_coulomb.cpp)
target_link_libraries(test_two_center_coulomb PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_two_center_coulomb)

# ERI erf-Coulomb kernel tests (Phase 7)
add_executable(test_eri_erf_coulomb unit/kernels/test_eri_erf_coulomb.cpp)
target_link_libraries(test_eri_erf_coulomb PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_eri_erf_coulomb)

# ERI erfc-Coulomb kernel tests (Phase 7)
add_executable(test_eri_erfc_coulomb unit/kernels/test_eri_erfc_coulomb.cpp)
target_link_libraries(test_eri_erfc_coulomb PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_eri_erfc_coulomb)

# norm_correction utility tests (Phase 7)
add_executable(test_norm_correction unit/kernels/test_norm_correction.cpp)
target_link_libraries(test_norm_correction PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_norm_correction)

# Engine tests
add_executable(test_engine unit/engine/test_engine.cpp)
target_link_libraries(test_engine PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_engine)

# Dispatch policy tests
add_executable(test_dispatch_policy unit/engine/test_dispatch_policy.cpp)
target_link_libraries(test_dispatch_policy PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_dispatch_policy)

# Unified Engine compute() API tests
add_executable(test_engine_compute unit/engine/test_engine_compute.cpp)
target_link_libraries(test_engine_compute PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_engine_compute)

# Full-basis 1e ShellSet driver regression tests
add_executable(test_fullbasis_1e_shellset_driver unit/engine/test_fullbasis_1e_shellset_driver.cpp)
target_link_libraries(test_fullbasis_1e_shellset_driver PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_fullbasis_1e_shellset_driver)

# Full-basis 2e ShellSet driver regression tests
add_executable(test_fullbasis_2e_shellset_driver unit/engine/test_fullbasis_2e_shellset_driver.cpp)
target_link_libraries(test_fullbasis_2e_shellset_driver PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_fullbasis_2e_shellset_driver)

# Dispatch registry and auto-tuning tests
add_executable(test_dispatch_registry unit/engine/test_dispatch_registry.cpp)
target_link_libraries(test_dispatch_registry PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_dispatch_registry)

# IntegralBuffer tests
add_executable(test_integral_buffer unit/engine/test_integral_buffer.cpp)
target_link_libraries(test_integral_buffer PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_integral_buffer)

# Engine allocation / O(N^4) fix verification tests
add_executable(test_engine_allocation unit/engine/test_engine_allocation.cpp)
target_link_libraries(test_engine_allocation PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_engine_allocation)

# Compute-and-consume pattern tests
add_executable(test_compute_and_consume unit/engine/test_compute_and_consume.cpp)
target_link_libraries(test_compute_and_consume PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_compute_and_consume)

# MultiGPU engine tests
add_executable(test_multi_gpu_engine unit/engine/test_multi_gpu_engine.cpp)
target_link_libraries(test_multi_gpu_engine PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_multi_gpu_engine PROPERTIES LABELS "gpu;experimental")

# Resource-aware dispatch tests (Step 10.4)
add_executable(test_resource_dispatch unit/engine/test_resource_dispatch.cpp)
target_link_libraries(test_resource_dispatch PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_resource_dispatch)

# Matrix assembly tests
add_executable(test_matrix_assembly unit/utils/test_matrix_assembly.cpp)
target_link_libraries(test_matrix_assembly PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_matrix_assembly)

# Parallel Fock correctness tests
add_executable(test_parallel_fock unit/utils/test_parallel_fock.cpp)
target_link_libraries(test_parallel_fock PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_parallel_fock)

# Device memory tests (CUDA)
add_executable(test_device_memory unit/memory/test_device_memory.cpp)
target_link_libraries(test_device_memory PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_device_memory PROPERTIES LABELS "gpu")

# Stream management tests (CUDA)
add_executable(test_stream_management unit/memory/test_stream_management.cpp)
target_link_libraries(test_stream_management PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_stream_management PROPERTIES LABELS "gpu")

# GPU execution slot tests (CUDA)
add_executable(test_gpu_execution_slot unit/memory/test_gpu_execution_slot.cpp)
target_link_libraries(test_gpu_execution_slot PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_gpu_execution_slot PROPERTIES LABELS "gpu")

# Device Boys function tests (CUDA)
add_executable(test_boys_device unit/cuda/test_boys_device.cpp)
target_link_libraries(test_boys_device PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_boys_device PROPERTIES LABELS "gpu")

# Device data upload tests (CUDA)
add_executable(test_device_data unit/cuda/test_device_data.cpp)
target_link_libraries(test_device_data PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_device_data PROPERTIES LABELS "gpu")

# CUDA overlap kernel tests
add_executable(test_overlap_kernel_cuda unit/cuda/test_overlap_kernel_cuda.cpp)
target_link_libraries(test_overlap_kernel_cuda PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_overlap_kernel_cuda PROPERTIES LABELS "gpu")

# CUDA kinetic kernel tests
add_executable(test_kinetic_kernel_cuda unit/cuda/test_kinetic_kernel_cuda.cpp)
target_link_libraries(test_kinetic_kernel_cuda PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_kinetic_kernel_cuda PROPERTIES LABELS "gpu")

# CUDA nuclear kernel tests
add_executable(test_nuclear_kernel_cuda unit/cuda/test_nuclear_kernel_cuda.cpp)
target_link_libraries(test_nuclear_kernel_cuda PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_nuclear_kernel_cuda PROPERTIES LABELS "gpu")

# CUDA ERI kernel tests
add_executable(test_eri_kernel_cuda unit/cuda/test_eri_kernel_cuda.cpp)
target_link_libraries(test_eri_kernel_cuda PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_eri_kernel_cuda PROPERTIES LABELS "gpu")

# CUDA Engine tests
add_executable(test_cuda_engine unit/cuda/test_cuda_engine.cpp)
target_link_libraries(test_cuda_engine PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_cuda_engine PROPERTIES LABELS "gpu")

# GPU FockBuilder tests
add_executable(test_fock_builder_gpu unit/cuda/test_fock_builder_gpu.cpp)
target_link_libraries(test_fock_builder_gpu PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_fock_builder_gpu PROPERTIES LABELS "gpu")

# =============================================================================
# Phase 12: Device Module Remediation Tests
# =============================================================================

# DeviceWorkspace and MultiDeviceMemoryManager tests (Tasks 12.3.1, 12.3.2, 12.3.4, 12.3.5)
add_executable(test_device_memory_manager unit/device/test_device_memory_manager.cpp)
target_link_libraries(test_device_memory_manager PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_device_memory_manager PROPERTIES LABELS "gpu")

# GPU stream handle tests (Task 12.3.3)
add_executable(test_stream_handle unit/device/test_stream_handle.cpp)
target_link_libraries(test_stream_handle PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_stream_handle PROPERTIES LABELS "gpu")

# Device-host transfer tests (Task 12.3.9)
add_executable(test_device_host_transfer unit/device/test_device_host_transfer.cpp)
target_link_libraries(test_device_host_transfer PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_device_host_transfer PROPERTIES LABELS "gpu")

# GPU error handling tests (Task 12.3.10)
add_executable(test_gpu_error_handling unit/device/test_gpu_error_handling.cpp)
target_link_libraries(test_gpu_error_handling PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_gpu_error_handling PROPERTIES LABELS "gpu")

# GPU compatibility layer tests (Task 12.1.2)
add_executable(test_gpu_compat unit/device/test_gpu_compat.cpp)
target_link_libraries(test_gpu_compat PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_gpu_compat PROPERTIES LABELS "gpu")

# =============================================================================
# Phase 13: MPI Module Remediation Tests
# =============================================================================

# MPI task distribution tests (Task 13.3.1)
add_executable(test_mpi_task_distribution unit/mpi/test_mpi_task_distribution.cpp)
target_link_libraries(test_mpi_task_distribution PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_mpi_task_distribution)

# MPI Fock builder tests (Task 13.3.2)
add_executable(test_mpi_fock_builder unit/mpi/test_mpi_fock_builder.cpp)
target_link_libraries(test_mpi_fock_builder PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_mpi_fock_builder)

# MPI timing statistics tests (Task 13.3.3)
add_executable(test_mpi_timing_statistics unit/mpi/test_mpi_timing_statistics.cpp)
target_link_libraries(test_mpi_timing_statistics PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_mpi_timing_statistics)

# MPI data serialization tests (Task 13.3.4)
add_executable(test_mpi_data_serialization unit/mpi/test_mpi_data_serialization.cpp)
target_link_libraries(test_mpi_data_serialization PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_mpi_data_serialization)

# MPI error handling tests (Task 13.3.5)
add_executable(test_mpi_error_handling unit/mpi/test_mpi_error_handling.cpp)
target_link_libraries(test_mpi_error_handling PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_mpi_error_handling)

# MPI integration Fock tests (Task 13.3.6)
add_executable(test_mpi_integration_fock unit/mpi/test_mpi_integration_fock.cpp)
target_link_libraries(test_mpi_integration_fock PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_mpi_integration_fock)

# -----------------------------------------------------------------------------
# Multi-rank MPI CTest definitions (Phase 5, Task 5.4)
#
# These tests exercise the real (non-stub) MPI code paths by launching each
# test executable under mpirun -np 2. They are only registered when the MPI
# backend is enabled at configure time.
# -----------------------------------------------------------------------------
if(LIBACCINT_USE_MPI)
    if(MPIEXEC_EXECUTABLE)
        # Multi-rank task distribution test
        add_test(
            NAME mpi_np2_task_distribution
            COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_mpi_task_distribution>
        )
        set_tests_properties(mpi_np2_task_distribution PROPERTIES
            LABELS "mpi"
            TIMEOUT 60
        )

        # Multi-rank Fock builder test
        add_test(
            NAME mpi_np2_fock_builder
            COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_mpi_fock_builder>
        )
        set_tests_properties(mpi_np2_fock_builder PROPERTIES
            LABELS "mpi"
            TIMEOUT 60
        )

        # Multi-rank integration Fock test
        add_test(
            NAME mpi_np2_integration_fock
            COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_mpi_integration_fock>
        )
        set_tests_properties(mpi_np2_integration_fock PROPERTIES
            LABELS "mpi"
            TIMEOUT 120
        )

        # Multi-rank error handling test
        add_test(
            NAME mpi_np2_error_handling
            COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_mpi_error_handling>
        )
        set_tests_properties(mpi_np2_error_handling PROPERTIES
            LABELS "mpi"
            TIMEOUT 60
        )

        # Multi-rank data serialization test
        add_test(
            NAME mpi_np2_data_serialization
            COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_mpi_data_serialization>
        )
        set_tests_properties(mpi_np2_data_serialization PROPERTIES
            LABELS "mpi"
            TIMEOUT 60
        )

        # Multi-rank timing statistics test
        add_test(
            NAME mpi_np2_timing_statistics
            COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_mpi_timing_statistics>
        )
        set_tests_properties(mpi_np2_timing_statistics PROPERTIES
            LABELS "mpi"
            TIMEOUT 60
        )
    else()
        message(WARNING "LIBACCINT_USE_MPI is ON but mpirun not found; skipping multi-rank CTests")
    endif()
endif()

# =============================================================================
# Phase 4.5: True Batched Execution Tests
# =============================================================================

# CPU parallel ShellSet tests
add_executable(test_shellset_parallel_cpu unit/engine/test_shellset_parallel_cpu.cpp)
target_link_libraries(test_shellset_parallel_cpu PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_shellset_parallel_cpu)

# CUDA batched ShellSetPair tests (1e integrals)
add_executable(test_shellsetpair_batched_1e unit/cuda/test_shellsetpair_batched_1e.cpp)
target_link_libraries(test_shellsetpair_batched_1e PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_shellsetpair_batched_1e PROPERTIES LABELS "gpu")

# CUDA batched ShellSetQuartet + Fock tests
add_executable(test_shellsetquartet_batched_fock unit/cuda/test_shellsetquartet_batched_fock.cpp)
target_link_libraries(test_shellsetquartet_batched_fock PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_shellsetquartet_batched_fock PROPERTIES LABELS "gpu")

# =============================================================================
# Phase 0.5: Engine GPU routing regression tests
# =============================================================================

# Engine-level GPU routing tests (verifies Engine::compute() dispatches to GPU)
add_executable(test_engine_gpu_routing unit/cuda/test_engine_gpu_routing.cpp)
target_link_libraries(test_engine_gpu_routing PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_engine_gpu_routing PROPERTIES LABELS "gpu")

# =============================================================================
# Phase 2: Screening Tests
# =============================================================================

# Schwarz bound tests
add_executable(test_schwarz_bound unit/screening/test_schwarz_bound.cpp)
target_link_libraries(test_schwarz_bound PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_schwarz_bound)

# Screening options tests
add_executable(test_screening_options unit/screening/test_screening_options.cpp)
target_link_libraries(test_screening_options PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_screening_options)

# Screened quartet iterator tests
add_executable(test_screened_iterator unit/screening/test_screened_iterator.cpp)
target_link_libraries(test_screened_iterator PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_screened_iterator)

# Density screening tests
add_executable(test_density_screening unit/screening/test_density_screening.cpp)
target_link_libraries(test_density_screening PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_density_screening)

# Screened Fock validation tests (Phase 2 Quality Gate)
add_executable(test_screened_fock_validation unit/screening/test_screened_fock_validation.cpp)
target_link_libraries(test_screened_fock_validation PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_screened_fock_validation)

# Schwarz bounds thread-safety tests (Phase 8 Tasks 8.3.1, 8.3.5)
add_executable(test_schwarz_thread_safety unit/screening/test_schwarz_thread_safety.cpp)
target_link_libraries(test_schwarz_thread_safety PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_schwarz_thread_safety)

# Symmetry exploitation and density screening tests (Phase 8 Tasks 8.3.2-8.3.4)
add_executable(test_screening_symmetry unit/screening/test_screening_symmetry.cpp)
target_link_libraries(test_screening_symmetry PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_screening_symmetry)

# =============================================================================
# Integration Tests
# =============================================================================

# One-electron integral end-to-end validation
add_executable(test_one_electron integration/test_one_electron.cpp)
target_link_libraries(test_one_electron PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_one_electron)

# Two-electron integral validation
add_executable(test_two_electron integration/test_two_electron.cpp)
target_link_libraries(test_two_electron PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_two_electron)

# HF energy calculation test (Quality Gate G2)
add_executable(test_hf_energy integration/test_hf_energy.cpp)
target_link_libraries(test_hf_energy PRIVATE
    libaccint
    GTest::gtest_main
    Eigen3::Eigen
)
gtest_discover_tests(test_hf_energy)

# GPU Correctness validation (Quality Gate G3)
add_executable(test_gpu_correctness integration/test_gpu_correctness.cpp)
target_link_libraries(test_gpu_correctness PRIVATE
    libaccint
    GTest::gtest_main
    Eigen3::Eigen
)
gtest_discover_tests(test_gpu_correctness PROPERTIES LABELS "gpu")

# D-shell (dd|dd) integral validation with PySCF reference
add_executable(test_dd_integrals integration/test_dd_integrals.cpp)
target_link_libraries(test_dd_integrals PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_dd_integrals)

# Higher AM (f-shell, g-shell) integral validation
add_executable(test_higher_am_integrals integration/test_higher_am.cpp)
target_link_libraries(test_higher_am_integrals PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_higher_am_integrals)

# High angular momentum integration tests with real basis sets (Step 13.5)
add_executable(test_high_am_integrals integration/test_high_am_integrals.cpp)
target_link_libraries(test_high_am_integrals PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_high_am_integrals PROPERTIES TIMEOUT 600)

# Batched one-electron workflow tests (Step 11.2)
add_executable(test_batched_1e_workflow integration/test_batched_1e_workflow.cpp)
target_link_libraries(test_batched_1e_workflow PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_batched_1e_workflow)

# Batched two-electron workflow tests (Step 11.3)
add_executable(test_batched_2e_workflow integration/test_batched_2e_workflow.cpp)
target_link_libraries(test_batched_2e_workflow PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_batched_2e_workflow)

# Batch size distribution analysis (Step 11.4)
add_executable(test_batch_statistics integration/test_batch_statistics.cpp)
target_link_libraries(test_batch_statistics PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_batch_statistics)

# Pipelined ERI computation tests (Step 11.5)
add_executable(test_pipelined_eri integration/test_pipelined_eri.cpp)
target_link_libraries(test_pipelined_eri PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_pipelined_eri)

# =============================================================================
# Phase 3: Range-Separated Operators Tests
# =============================================================================

# Range-separated Boys function tests
add_executable(test_boys_range_separated unit/math/test_boys_range_separated.cpp)
target_link_libraries(test_boys_range_separated PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_boys_range_separated)

# Range-separated ERI decomposition tests (Quality Gate G3)
add_executable(test_range_separated_eri integration/test_range_separated_eri.cpp)
target_link_libraries(test_range_separated_eri PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_range_separated_eri)

# Phase 17: Property integral tests
add_executable(test_multipole_kernel unit/kernels/test_multipole_kernel.cpp)
target_link_libraries(test_multipole_kernel PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_multipole_kernel)

add_executable(test_momentum_kernel unit/kernels/test_momentum_kernel.cpp)
target_link_libraries(test_momentum_kernel PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_momentum_kernel)

add_executable(test_distributed_multipole_kernel unit/kernels/test_distributed_multipole_kernel.cpp)
target_link_libraries(test_distributed_multipole_kernel PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_distributed_multipole_kernel)

add_executable(test_interaction_tensors unit/math/test_interaction_tensors.cpp)
target_link_libraries(test_interaction_tensors PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_interaction_tensors)

# Phase 3 Math Remediation: New test targets
add_executable(test_cholesky unit/math/test_cholesky.cpp)
target_link_libraries(test_cholesky PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_cholesky)

add_executable(test_math_constants unit/math/test_math_constants.cpp)
target_link_libraries(test_math_constants PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_math_constants)

add_executable(test_projection_operator unit/operators/test_projection_operator.cpp)
target_link_libraries(test_projection_operator PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_projection_operator)

# =============================================================================
# Phase 5: Operators Module Remediation Tests
# =============================================================================

# Property integral classification tests (Task 5.3.1)
add_executable(test_property_integral_classification unit/operators/test_property_integrals.cpp)
target_link_libraries(test_property_integral_classification PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_property_integral_classification)

# Device operator data tests (Task 5.3.5 — DISABLED for CPU builds)
add_executable(test_device_operator_data unit/operators/test_device_operator_data.cpp)
target_link_libraries(test_device_operator_data PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_device_operator_data PROPERTIES LABELS "gpu")

add_executable(test_multi_component_buffer unit/engine/test_multi_component_buffer.cpp)
target_link_libraries(test_multi_component_buffer PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_multi_component_buffer)

add_executable(test_property_integrals integration/test_property_integrals.cpp)
target_link_libraries(test_property_integrals PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_property_integrals)

# =============================================================================
# Phase 18: Thread-Parallel CPU Execution Tests
# =============================================================================

# Thread configuration tests
add_executable(test_thread_config unit/engine/test_thread_config.cpp)
target_link_libraries(test_thread_config PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_thread_config)

# Thread-local pool tests
add_executable(test_thread_local_pool unit/memory/test_thread_local_pool.cpp)
target_link_libraries(test_thread_local_pool PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_thread_local_pool)

# Parallel 1e vs serial comparison tests
add_executable(test_parallel_1e unit/engine/test_parallel_1e.cpp)
target_link_libraries(test_parallel_1e PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_parallel_1e)

# FockBuilder unit tests (Phase 10)
add_executable(test_fock_builder unit/consumers/test_fock_builder.cpp)
target_link_libraries(test_fock_builder PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_fock_builder)

# DFFockBuilder UHF exchange tests (Phase 10)
add_executable(test_df_fock_builder_uhf unit/consumers/test_df_fock_builder_uhf.cpp)
target_link_libraries(test_df_fock_builder_uhf PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_df_fock_builder_uhf)

# DFFockBuilder unit tests
add_executable(test_df_fock_builder unit/consumers/test_df_fock_builder.cpp)
target_link_libraries(test_df_fock_builder PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_df_fock_builder)

# Consumer accumulate pattern tests (Phase 10)
add_executable(test_consumer_accumulate unit/consumers/test_consumer_accumulate.cpp)
target_link_libraries(test_consumer_accumulate PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_consumer_accumulate)

# Parallel scaling integration tests
add_executable(test_parallel_scaling integration/test_parallel_scaling.cpp)
target_link_libraries(test_parallel_scaling PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_parallel_scaling)

# Screening end-to-end integration tests (Phase 8)
add_executable(test_screening_integration integration/test_screening_integration.cpp)
target_link_libraries(test_screening_integration PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_screening_integration)

# Spherical integral integration tests
add_executable(test_spherical_integrals integration/test_spherical_integrals.cpp)
target_link_libraries(test_spherical_integrals PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_spherical_integrals)

# =============================================================================
# Phase 22: Density Fitting Completion Tests
# =============================================================================

# Auxiliary basis parser tests (Task 22.1.1)
add_executable(test_auxiliary_basis_parser unit/df/test_auxiliary_basis_parser.cpp)
target_link_libraries(test_auxiliary_basis_parser PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_auxiliary_basis_parser)

# Auxiliary basis data tests (Task 22.1.2)
add_executable(test_auxiliary_basis_data unit/df/test_auxiliary_basis_data.cpp)
target_link_libraries(test_auxiliary_basis_data PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_auxiliary_basis_data)

# Auxiliary basis selector tests (Task 22.1.3)
add_executable(test_auxiliary_basis_selector unit/df/test_auxiliary_basis_selector.cpp)
target_link_libraries(test_auxiliary_basis_selector PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_auxiliary_basis_selector)

# Three-center block storage tests (Task 22.2.1)
add_executable(test_three_center_storage unit/df/test_three_center_storage.cpp)
target_link_libraries(test_three_center_storage PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_three_center_storage)

# Out-of-core storage tests (Task 22.2.2)
add_executable(test_out_of_core_storage unit/df/test_out_of_core_storage.cpp)
target_link_libraries(test_out_of_core_storage PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_out_of_core_storage)

# DF-Fock validation tests (Task 22.3.1)
add_executable(test_df_fock_validation validation/test_df_fock_validation.cpp)
target_link_libraries(test_df_fock_validation PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_df_fock_validation)

# DF-HF energy validation tests (Task 22.3.2)
add_executable(test_df_hf_energy validation/test_df_hf_energy.cpp)
target_link_libraries(test_df_hf_energy PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_df_hf_energy)

# DF error analysis tests (Task 22.3.3)
add_executable(test_df_error_analysis validation/test_df_error_analysis.cpp)
target_link_libraries(test_df_error_analysis PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_df_error_analysis)

# DF GPU pipeline tests (Task 22.4.3)
add_executable(test_df_gpu_pipeline unit/df/test_df_gpu_pipeline.cpp)
target_link_libraries(test_df_gpu_pipeline PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_df_gpu_pipeline PROPERTIES LABELS "gpu")

# DF range-separated tests (Task 22.4.4)
add_executable(test_df_range_separated unit/df/test_df_range_separated.cpp)
target_link_libraries(test_df_range_separated PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_df_range_separated)

# =============================================================================
# Phase 11: DF Module Remediation Tests
# =============================================================================

# DFFockBuilder unit tests (Task 11.3.1)
add_executable(test_df_fock_builder_unit unit/df/test_df_fock_builder_unit.cpp)
target_link_libraries(test_df_fock_builder_unit PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_df_fock_builder_unit)

# BLAS contraction correctness tests (Task 11.3.2)
add_executable(test_df_blas_contractions unit/df/test_df_blas_contractions.cpp)
target_link_libraries(test_df_blas_contractions PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_df_blas_contractions)

# Cholesky conditioning tests (Task 11.3.3)
add_executable(test_df_cholesky_conditioning unit/df/test_df_cholesky_conditioning.cpp)
target_link_libraries(test_df_cholesky_conditioning PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_df_cholesky_conditioning)

# Storage integration tests (Task 11.3.4)
add_executable(test_df_storage_integration unit/df/test_df_storage_integration.cpp)
target_link_libraries(test_df_storage_integration PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_df_storage_integration)

# DF vs conventional comparison tests (Task 11.3.5)
add_executable(test_df_vs_conventional unit/df/test_df_vs_conventional.cpp)
target_link_libraries(test_df_vs_conventional PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_df_vs_conventional)

# UHF/RHF equivalence tests (Task 11.3.6)
add_executable(test_df_fock_builder_uhf_rhf unit/df/test_df_fock_builder_uhf_rhf.cpp)
target_link_libraries(test_df_fock_builder_uhf_rhf PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_df_fock_builder_uhf_rhf)

# =============================================================================
# Phase 23: Higher AM Validation
# =============================================================================

# Higher AM integral validation (Tasks 23.1.1-23.1.3)
add_executable(test_higher_am_validation validation/test_higher_am_validation.cpp)
target_link_libraries(test_higher_am_validation PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_higher_am_validation)

# Higher AM GPU validation (Task 23.1.4)
add_executable(test_higher_am_gpu validation/test_higher_am_gpu.cpp)
target_link_libraries(test_higher_am_gpu PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_higher_am_gpu PROPERTIES LABELS "gpu")

# =============================================================================
# Phase 24: Single Precision Pipeline
# =============================================================================

# Float32 vs Float64 error analysis (Task 24.4.1)
add_executable(test_float32_accuracy validation/test_float32_accuracy.cpp)
target_link_libraries(test_float32_accuracy PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_float32_accuracy)

# Float32 HF energy validation (Task 24.4.2)
add_executable(test_float32_hf_energy validation/test_float32_hf_energy.cpp)
target_link_libraries(test_float32_hf_energy PRIVATE libaccint GTest::gtest_main Eigen3::Eigen)
gtest_discover_tests(test_float32_hf_energy)

# Float32 GPU speedup measurement (Task 24.4.3)
add_executable(test_float32_gpu_speedup validation/test_float32_gpu_speedup.cpp)
target_link_libraries(test_float32_gpu_speedup PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_float32_gpu_speedup PROPERTIES LABELS "gpu")

# Float32 Boys function validation (Task 24.2.3)
add_executable(test_float32_boys validation/test_float32_boys.cpp)
target_link_libraries(test_float32_boys PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_float32_boys)

# Float32 CPU kernel validation (Tasks 24.2.1, 24.2.2)
add_executable(test_float32_kernels validation/test_float32_kernels.cpp)
target_link_libraries(test_float32_kernels PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_float32_kernels)

# Mixed precision accumulation tests (Task 24.3.3)
add_executable(test_mixed_precision validation/test_mixed_precision.cpp)
target_link_libraries(test_mixed_precision PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_mixed_precision)

# Multi-GPU correctness validation tests
add_executable(test_multi_gpu_correctness validation/test_multi_gpu_correctness.cpp)
target_link_libraries(test_multi_gpu_correctness PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_multi_gpu_correctness PROPERTIES LABELS "gpu;experimental")

# Phase 16 — Integration validation and regression tests
add_executable(test_integration_validation validation/test_integration_validation.cpp)
target_link_libraries(test_integration_validation PRIVATE
    libaccint
    GTest::gtest_main
    Eigen3::Eigen
)
gtest_discover_tests(test_integration_validation)

# Phase 17 — Comprehensive validation (all mol/basis vs PySCF)
add_executable(test_comprehensive_validation validation/test_comprehensive_validation.cpp)
target_link_libraries(test_comprehensive_validation PRIVATE libaccint GTest::gtest_main Eigen3::Eigen)
gtest_discover_tests(test_comprehensive_validation PROPERTIES TIMEOUT 600)

# =============================================================================
# Benchmarks (optional)
# =============================================================================

if(LIBACCINT_BUILD_BENCHMARKS)
    # GPU vs CPU benchmark suite
    add_executable(bench_gpu_integrals benchmark/bench_gpu_integrals.cpp)
    target_link_libraries(bench_gpu_integrals PRIVATE
        libaccint
        benchmark::benchmark
    )

    # Boys function benchmark (scalar vs SIMD)
    add_executable(bench_boys benchmark/bench_boys.cpp)
    target_link_libraries(bench_boys PRIVATE
        libaccint
        benchmark::benchmark
    )

    # Fock build benchmark (threading comparison)
    add_executable(bench_fock_build benchmark/bench_fock_build.cpp)
    target_link_libraries(bench_fock_build PRIVATE
        libaccint
        benchmark::benchmark
    )

    # Thread scaling benchmark
    add_executable(bench_scaling benchmark/bench_scaling.cpp)
    target_link_libraries(bench_scaling PRIVATE
        libaccint
        benchmark::benchmark
    )

    # Phase 27: Per-kernel microbenchmarks (Task 27.1.3)
    add_executable(bench_kernel_microbenchmarks benchmark/bench_kernel_microbenchmarks.cpp)
    target_link_libraries(bench_kernel_microbenchmarks PRIVATE
        libaccint
        benchmark::benchmark
    )

    # Phase 27: End-to-end HF benchmark (Task 27.1.4)
    add_executable(bench_hf_endtoend benchmark/bench_hf_endtoend.cpp)
    target_link_libraries(bench_hf_endtoend PRIVATE
        libaccint
        benchmark::benchmark
    )

    # Phase 27: CPU optimization profiling (Tasks 27.2.1–27.2.4)
    add_executable(bench_cpu_optimization benchmark/bench_cpu_optimization.cpp)
    target_link_libraries(bench_cpu_optimization PRIVATE
        libaccint
        benchmark::benchmark
    )

    # Phase 27: GPU optimization profiling (Tasks 27.3.1–27.3.3)
    add_executable(bench_gpu_optimization benchmark/bench_gpu_optimization.cpp)
    target_link_libraries(bench_gpu_optimization PRIVATE
        libaccint
        benchmark::benchmark
    )

    # DF performance benchmarks (Tasks 22.4.1–22.4.2)
    add_executable(bench_df_performance benchmark/bench_df_performance.cpp)
    target_link_libraries(bench_df_performance PRIVATE
        libaccint
        benchmark::benchmark
    )

    # Multi-GPU scaling benchmarks (CUDA only)
    if(LIBACCINT_USE_CUDA)
        add_executable(bench_multi_gpu_scaling benchmark/bench_multi_gpu_scaling.cpp)
        target_link_libraries(bench_multi_gpu_scaling PRIVATE
            libaccint
            benchmark::benchmark
        )
    endif()

    # Screening performance benchmarks (Task 15.3.1)
    add_executable(bench_screening benchmark/bench_screening.cpp)
    target_link_libraries(bench_screening PRIVATE
        libaccint
        benchmark::benchmark
    )

    # Memory pool benchmarks (Task 15.3.2)
    add_executable(bench_memory_pool benchmark/bench_memory_pool.cpp)
    target_link_libraries(bench_memory_pool PRIVATE
        libaccint
        benchmark::benchmark
    )

    # Dispatch overhead benchmarks (Task 15.3.3)
    add_executable(bench_dispatch_overhead benchmark/bench_dispatch_overhead.cpp)
    target_link_libraries(bench_dispatch_overhead PRIVATE
        libaccint
        benchmark::benchmark
    )

    # Phase 17 — Comprehensive benchmarks (all mol/basis combos)
    add_executable(bench_comprehensive benchmark/bench_comprehensive.cpp)
    target_link_libraries(bench_comprehensive PRIVATE libaccint benchmark::benchmark benchmark::benchmark_main)
endif()

# CPU vs GPU benchmark (standalone, no Google Benchmark dependency)
add_executable(bench_cpu_vs_gpu benchmark/bench_cpu_vs_gpu.cpp)
target_link_libraries(bench_cpu_vs_gpu PRIVATE libaccint)

# Optimal dispatch benchmark (standalone, CUDA only — benchmarks all variants)
add_executable(bench_optimal_dispatch benchmark/bench_optimal_dispatch.cpp)
target_link_libraries(bench_optimal_dispatch PRIVATE libaccint nlohmann_json::nlohmann_json)

# High-AM performance validation benchmark (Step 13.6 — GoogleTest, standalone)
add_executable(bench_high_am benchmark/bench_high_am.cpp)
target_link_libraries(bench_high_am PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(bench_high_am PROPERTIES TIMEOUT 600)

# =============================================================================
# Phase 25: Production Hardening Tests
# =============================================================================

# Logging infrastructure tests (Task 25.3.2)
add_executable(test_logging unit/utils/test_logging.cpp)
target_link_libraries(test_logging PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_logging)

# Diagnostics infrastructure tests (Task 25.3.3)
add_executable(test_diagnostics unit/utils/test_diagnostics.cpp)
target_link_libraries(test_diagnostics PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_diagnostics)

# Input validation tests (Task 25.4.1)
add_executable(test_input_validation unit/utils/test_input_validation.cpp)
target_link_libraries(test_input_validation PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_input_validation)

# Debug checks tests (Task 25.3.4)
add_executable(test_debug_checks unit/utils/test_debug_checks.cpp)
target_link_libraries(test_debug_checks PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_debug_checks)

# Edge case handling tests (Tasks 25.2.1, 25.2.3)
add_executable(test_edge_cases validation/test_edge_cases.cpp)
target_link_libraries(test_edge_cases PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_edge_cases)

# Large system stress tests (Task 25.2.2)
add_executable(test_large_system validation/test_large_system.cpp)
target_link_libraries(test_large_system PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_large_system)

# Sanitizer clean tests (Tasks 25.1.1-25.1.4)
add_executable(test_sanitizer_clean validation/test_sanitizer_clean.cpp)
target_link_libraries(test_sanitizer_clean PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_sanitizer_clean)

# Exception safety tests (Task 25.4.2)
add_executable(test_exception_safety validation/test_exception_safety.cpp)
target_link_libraries(test_exception_safety PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_exception_safety)

# Built-in basis set tests (Tasks 25.4.3, 25.4.4)
add_executable(test_basis_expansion validation/test_basis_expansion.cpp)
target_link_libraries(test_basis_expansion PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_basis_expansion)

# =============================================================================
# Phase 26: CI/CD & Release Packaging Tests
# =============================================================================

# Version API tests (Task 26.4.1)
add_executable(test_version validation/test_version.cpp)
target_link_libraries(test_version PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_version PROPERTIES LABELS "release")

# Install integration smoke tests (Task 26.3.1)
add_executable(test_install_integration validation/test_install_integration.cpp)
target_link_libraries(test_install_integration PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_install_integration PROPERTIES LABELS "release")

# =============================================================================
# Phase 27: Performance Benchmarking & Optimization Tests
# =============================================================================

# CPU optimization validation tests (Tasks 27.2.1–27.2.4)
add_executable(test_cpu_optimization validation/test_cpu_optimization.cpp)
target_link_libraries(test_cpu_optimization PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_cpu_optimization)

# GPU optimization validation tests (Tasks 27.3.1–27.3.3)
add_executable(test_gpu_optimization validation/test_gpu_optimization.cpp)
target_link_libraries(test_gpu_optimization PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_gpu_optimization PROPERTIES LABELS "gpu")

# Phase 28: API Finalization, Examples & Documentation
# BSE JSON parser validation tests (Task 28.2.1)
add_executable(test_bse_parser validation/test_bse_parser.cpp)
target_link_libraries(test_bse_parser PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_bse_parser)

# Example pattern validation tests (Tasks 28.1.1–28.1.6)
add_executable(test_examples validation/test_examples.cpp)
target_link_libraries(test_examples PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_examples)

# =============================================================================
# Phase 1: Core & Utils Remediation Tests
# =============================================================================

# Aligned allocation tests (Task 1.3.1)
add_executable(test_aligned_alloc unit/utils/test_aligned_alloc.cpp)
target_link_libraries(test_aligned_alloc PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_aligned_alloc)

# SIMD abstraction tests (Task 1.3.2)
add_executable(test_simd unit/utils/test_simd.cpp)
target_link_libraries(test_simd PRIVATE libaccint GTest::gtest_main)
# Apply the same SIMD flags that the library uses so the header compiles correctly
if(LIBACCINT_HAS_AVX512)
    target_compile_options(test_simd PRIVATE -mavx512f -mavx512dq)
elseif(LIBACCINT_HAS_AVX2)
    target_compile_options(test_simd PRIVATE -mavx2 -mfma)
endif()
gtest_discover_tests(test_simd)

# TIMED_SCOPE macro tests (Task 1.3.3)
add_executable(test_timed_scope unit/utils/test_timed_scope.cpp)
target_link_libraries(test_timed_scope PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_timed_scope)

# Diagnostics thread-safety tests (Task 1.3.4)
add_executable(test_diagnostics_thread_safety unit/utils/test_diagnostics_thread_safety.cpp)
target_link_libraries(test_diagnostics_thread_safety PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_diagnostics_thread_safety)

# Logger thread-safety tests (Task 1.3.5)
add_executable(test_logger_thread_safety unit/utils/test_logger_thread_safety.cpp)
target_link_libraries(test_logger_thread_safety PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_logger_thread_safety)

# Spherical transform utility tests (Task 1.3.6)
add_executable(test_spherical_transforms unit/utils/test_spherical_transforms.cpp)
target_link_libraries(test_spherical_transforms PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_spherical_transforms)

# =============================================================================
# Phase 2: Memory Remediation Tests
# =============================================================================

# MemoryPool, GlobalMemoryPool, PooledBuffer tests (Tasks 2.3.1, 2.3.3, 2.3.4, 2.3.6)
add_executable(test_memory_pool unit/memory/test_memory_pool.cpp)
target_link_libraries(test_memory_pool PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_memory_pool)

# BatchBufferPool tests (Task 2.3.2)
add_executable(test_buffer_pool unit/memory/test_buffer_pool.cpp)
target_link_libraries(test_buffer_pool PRIVATE libaccint GTest::gtest_main)
gtest_discover_tests(test_buffer_pool)

# =============================================================================
# Numerical Golden-Value Regression Tests (Step 9.4)
# =============================================================================

# Regression tests verifying numerical correctness against physical constraints
add_executable(test_numerical_golden_values regression/test_numerical_golden_values.cpp)
target_link_libraries(test_numerical_golden_values PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_numerical_golden_values)

# =============================================================================
# Build Pipeline Validation
# =============================================================================

# Lightweight shell-based smoke test for the handwritten-only build pipeline
add_test(
    NAME build_pipeline_validation
    COMMAND ${CMAKE_SOURCE_DIR}/tests/test_build_pipeline.sh ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}
)

# =============================================================================
# ERI Tensor API Tests (Step 14.7)
# =============================================================================

# Non-consumer ERI API: compute() overloads, compute_eri_tensor/block, screening, parallel
add_executable(test_eri_tensor_api integration/test_eri_tensor_api.cpp)
target_link_libraries(test_eri_tensor_api PRIVATE
    libaccint
    GTest::gtest_main
)
gtest_discover_tests(test_eri_tensor_api)

# Propagate AddressSanitizer flags to test executables so they link ASan
# before the shared library (required by GCC's ASan runtime).
get_property(_libaccint_test_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
if(LIBACCINT_USE_ASAN AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    foreach(_tgt IN LISTS _libaccint_test_targets)
        get_target_property(_tgt_type "${_tgt}" TYPE)
        if(_tgt_type STREQUAL "EXECUTABLE")
            target_compile_options(${_tgt} PRIVATE -fsanitize=address -fsanitize=undefined)
            target_link_options(${_tgt} PRIVATE -fsanitize=address -fsanitize=undefined)
        endif()
    endforeach()
endif()

# Ensure ctest -N remains clean when only a subset of test executables is built.
# gtest_discover_tests() creates *_include.cmake wrappers that emit *_NOT_BUILT
# placeholder tests when the corresponding *_tests.cmake manifest is absent.
# Pre-creating empty manifests avoids placeholder noise in partial-build workflows.
foreach(_tgt IN LISTS _libaccint_test_targets)
    get_target_property(_tgt_type "${_tgt}" TYPE)
    if(_tgt_type STREQUAL "EXECUTABLE")
        set(_gtest_manifest "${CMAKE_CURRENT_BINARY_DIR}/${_tgt}[1]_tests.cmake")
        if(NOT EXISTS "${_gtest_manifest}")
            file(WRITE "${_gtest_manifest}"
                "# Pre-created by tests/CMakeLists.txt to suppress _NOT_BUILT placeholders.\n")
        endif()
    endif()
endforeach()
