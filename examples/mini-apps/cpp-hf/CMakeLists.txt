cmake_minimum_required(VERSION 3.20)
project(hf_miniapp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try find_package first, then fall back to parent project
find_package(LibAccInt QUIET)
if(NOT LibAccInt_FOUND)
    # When building as part of libaccint source tree
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../CMakeLists.txt")
        # Add parent project
        add_subdirectory(
            ${CMAKE_CURRENT_SOURCE_DIR}/../../..
            ${CMAKE_BINARY_DIR}/libaccint
            EXCLUDE_FROM_ALL
        )
    else()
        message(FATAL_ERROR "LibAccInt not found. Install it or build from the source tree.")
    endif()
endif()

add_executable(hf_miniapp main.cpp)

# Support both installed-package and in-tree target names.
if(TARGET LibAccInt::libaccint)
    target_link_libraries(hf_miniapp PRIVATE LibAccInt::libaccint)
elseif(TARGET libaccint::libaccint)
    target_link_libraries(hf_miniapp PRIVATE libaccint::libaccint)
elseif(TARGET libaccint)
    target_link_libraries(hf_miniapp PRIVATE libaccint)
else()
    message(FATAL_ERROR "Could not find a LibAccInt link target (expected one of: LibAccInt::libaccint, libaccint::libaccint, libaccint)")
endif()
