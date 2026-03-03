# cmake/abi-version.cmake — ABI version management and symbol versioning
# Task 26.4.3: ABI stability policy and symbol versioning
#
# Manages SO version numbers and optionally applies linker version scripts
# for symbol visibility control on Linux.

# =============================================================================
# ABI Version Numbers
# =============================================================================
#
# ABI versioning follows the convention:
#   SO version = MAJOR (changes on ABI-breaking releases)
#   SO version = MAJOR.MINOR (full interface version)
#
# Policy:
#   - Major version 1.x: ABI may change between minor versions
#   - Major version 2+: ABI stable within major version; minor adds symbols
#
# The SO version is set from PROJECT_VERSION_MAJOR in the root CMakeLists.txt.
# Additional ABI tracking is provided by this script.

set(LIBACCINT_ABI_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(LIBACCINT_ABI_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(LIBACCINT_ABI_VERSION "${LIBACCINT_ABI_VERSION_MAJOR}.${LIBACCINT_ABI_VERSION_MINOR}")

message(STATUS "ABI version: ${LIBACCINT_ABI_VERSION} (SO major: ${LIBACCINT_ABI_VERSION_MAJOR})")

# =============================================================================
# Linker Version Script (Linux only)
# =============================================================================

function(apply_version_script target)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        return()
    endif()

    set(VERSION_SCRIPT "${CMAKE_SOURCE_DIR}/libaccint.map")
    if(NOT EXISTS "${VERSION_SCRIPT}")
        message(STATUS "No linker version script found — skipping symbol versioning for ${target}")
        return()
    endif()

    # Apply version script to control symbol visibility
    target_link_options(${target} PRIVATE
        "LINKER:--version-script=${VERSION_SCRIPT}"
    )

    # Mark the version script as a dependency so changes trigger re-link
    set_target_properties(${target} PROPERTIES
        LINK_DEPENDS "${VERSION_SCRIPT}"
    )

    message(STATUS "Applied version script to ${target}: ${VERSION_SCRIPT}")
endfunction()

# =============================================================================
# Symbol Visibility Defaults
# =============================================================================

function(apply_symbol_visibility target)
    # Hide symbols by default; only export explicitly marked symbols
    set_target_properties(${target} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endfunction()

# =============================================================================
# ABI Compatibility Check Target
# =============================================================================
#
# Usage: cmake --build . --target abi-check
# Requires: abigail-tools (abidiff, abidw)
#
# This target compares the current ABI against a baseline dump.

function(add_abi_check_target target)
    find_program(ABIDW_EXECUTABLE abidw)
    find_program(ABIDIFF_EXECUTABLE abidiff)

    if(NOT ABIDW_EXECUTABLE OR NOT ABIDIFF_EXECUTABLE)
        message(STATUS "abigail-tools not found — ABI check target disabled")
        return()
    endif()

    set(ABI_BASELINE "${CMAKE_SOURCE_DIR}/abi/baseline.abi.xml")
    set(ABI_CURRENT "${CMAKE_BINARY_DIR}/abi/current.abi.xml")

    # Target to dump current ABI
    add_custom_target(abi-dump
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/abi"
        COMMAND ${ABIDW_EXECUTABLE} $<TARGET_FILE:${target}> > "${ABI_CURRENT}"
        DEPENDS ${target}
        COMMENT "Dumping ABI for ${target}..."
    )

    # Target to compare ABI against baseline
    if(EXISTS "${ABI_BASELINE}")
        add_custom_target(abi-check
            COMMAND ${ABIDIFF_EXECUTABLE}
                --drop-private-types
                --no-unreferenced-symbols
                "${ABI_BASELINE}" "${ABI_CURRENT}"
            DEPENDS abi-dump
            COMMENT "Checking ABI compatibility..."
        )
    else()
        add_custom_target(abi-check
            COMMAND ${CMAKE_COMMAND} -E echo "No ABI baseline found at ${ABI_BASELINE}"
            COMMAND ${CMAKE_COMMAND} -E echo "Run 'cmake --build . --target abi-dump' to create baseline"
            DEPENDS abi-dump
        )
    endif()
endfunction()
