# =============================================================================
# LibAccInt source directory
# =============================================================================
#
# Source file listings are defined in the root CMakeLists.txt.
# This file handles supplementary source-level configuration:
#
#   1. Per-TU compilation properties for generated CUDA kernels
#   2. Conditional inclusion of generated_sources.cmake from codegen output
#   3. Additional source files added via target_sources()
#
# The generated_sources.cmake file (produced by the codegen system) uses
# target_sources() to append generated .cu files directly to the libaccint
# target, so no manual source listing is needed here for generated code.
# =============================================================================

# ---------------------------------------------------------------------------
# Device resource tracker — only compiled when CUDA is enabled
# ---------------------------------------------------------------------------
# Note: device_resource_tracker.cu is listed in the root CMakeLists.txt
# LIBACCINT_CUDA_SOURCES variable. This section is reserved for any
# per-file compile properties that generated or device sources may need.

# ---------------------------------------------------------------------------
# Per-TU compilation for generated CUDA kernels
# ---------------------------------------------------------------------------
# When generated_sources.cmake is included (from root CMakeLists.txt), it
# sets CUDA_SEPARABLE_COMPILATION ON for each generated .cu file.  This
# ensures every generated kernel compiles as its own translation unit,
# avoiding the unity-build memory bottleneck with nvcc.
#
# Additional per-file flags can be applied here if needed:
#
#   if(LIBACCINT_USE_CUDA AND LIBACCINT_GENERATED_CUDA_SOURCES)
#       set_source_files_properties(
#           ${LIBACCINT_GENERATED_CUDA_SOURCES}
#           PROPERTIES
#               CUDA_SEPARABLE_COMPILATION ON
#       )
#   endif()
