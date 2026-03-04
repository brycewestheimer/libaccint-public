# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

add_executable(shellset_batched_1e shellset_batched_1e.cpp)
target_link_libraries(shellset_batched_1e PRIVATE libaccint)

add_executable(shellset_batched_fock shellset_batched_fock.cpp)
target_link_libraries(shellset_batched_fock PRIVATE libaccint)

add_executable(basic_hf_energy basic_hf_energy.cpp)
target_link_libraries(basic_hf_energy PRIVATE libaccint)

add_executable(gpu_fock_build gpu_fock_build.cpp)
target_link_libraries(gpu_fock_build PRIVATE libaccint)

add_executable(qmm_embedding qmm_embedding.cpp)
target_link_libraries(qmm_embedding PRIVATE libaccint)

add_executable(density_fitting density_fitting.cpp)
target_link_libraries(density_fitting PRIVATE libaccint)

add_executable(custom_consumer custom_consumer.cpp)
target_link_libraries(custom_consumer PRIVATE libaccint)

add_executable(gpu_hf_workflow gpu_hf_workflow.cpp)
target_link_libraries(gpu_hf_workflow PRIVATE libaccint)

add_executable(hf_calculation hf_calculation.cpp)
target_link_libraries(hf_calculation PRIVATE libaccint Eigen3::Eigen)
