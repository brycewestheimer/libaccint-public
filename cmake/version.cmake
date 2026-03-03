# cmake/version.cmake — Version management from git tags
# Task 26.4.1: Semantic versioning automation
#
# Extracts version from git tags or falls back to project VERSION.
# Usage: include(version) after project() declaration.

# Try to get version from git
find_package(Git QUIET)

if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    # Get the latest tag matching vX.Y.Z
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --match "v[0-9]*" --abbrev=0
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE LIBACCINT_GIT_TAG
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE GIT_TAG_RESULT
    )

    # Get full git describe (includes commit count and hash)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --match "v[0-9]*" --long --dirty
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE LIBACCINT_GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE GIT_DESCRIBE_RESULT
    )

    # Get current commit hash
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short=8 HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE LIBACCINT_GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # Get branch name
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE LIBACCINT_GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(GIT_TAG_RESULT EQUAL 0)
        # Parse tag into components: v1.2.3 -> 1, 2, 3
        string(REGEX MATCH "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)" _match "${LIBACCINT_GIT_TAG}")
        if(_match)
            set(LIBACCINT_VERSION_FROM_GIT "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
            message(STATUS "Version from git tag: ${LIBACCINT_VERSION_FROM_GIT}")
        endif()
    endif()

    if(GIT_DESCRIBE_RESULT EQUAL 0)
        # Parse distance from tag: v1.2.3-5-g1234abcd-dirty
        string(REGEX MATCH "-([0-9]+)-g" _match "${LIBACCINT_GIT_DESCRIBE}")
        if(_match)
            set(LIBACCINT_GIT_DISTANCE "${CMAKE_MATCH_1}")
        else()
            set(LIBACCINT_GIT_DISTANCE "0")
        endif()

        if(LIBACCINT_GIT_DESCRIBE MATCHES "-dirty$")
            set(LIBACCINT_GIT_DIRTY TRUE)
        else()
            set(LIBACCINT_GIT_DIRTY FALSE)
        endif()

        message(STATUS "Git describe: ${LIBACCINT_GIT_DESCRIBE}")
        message(STATUS "Git commit: ${LIBACCINT_GIT_HASH}")
        message(STATUS "Git branch: ${LIBACCINT_GIT_BRANCH}")
    endif()
else()
    message(STATUS "Git not found or not a git repository — using project VERSION")
    set(LIBACCINT_GIT_TAG "")
    set(LIBACCINT_GIT_DESCRIBE "")
    set(LIBACCINT_GIT_HASH "unknown")
    set(LIBACCINT_GIT_BRANCH "unknown")
    set(LIBACCINT_GIT_DISTANCE "0")
    set(LIBACCINT_GIT_DIRTY FALSE)
endif()

# Use project version as canonical source
set(LIBACCINT_VERSION "${PROJECT_VERSION}")
set(LIBACCINT_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(LIBACCINT_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(LIBACCINT_VERSION_PATCH "${PROJECT_VERSION_PATCH}")

# Build version string incorporating pre-release suffix (if set) and git metadata
if(DEFINED LIBACCINT_PRERELEASE AND NOT LIBACCINT_PRERELEASE STREQUAL "")
    set(LIBACCINT_VERSION_FULL "${LIBACCINT_VERSION}-${LIBACCINT_PRERELEASE}")
else()
    set(LIBACCINT_VERSION_FULL "${LIBACCINT_VERSION}")
endif()

if(LIBACCINT_GIT_DISTANCE GREATER 0)
    string(APPEND LIBACCINT_VERSION_FULL "+dev.${LIBACCINT_GIT_DISTANCE}.${LIBACCINT_GIT_HASH}")
endif()

if(LIBACCINT_GIT_DIRTY)
    string(APPEND LIBACCINT_VERSION_FULL ".dirty")
endif()

message(STATUS "LibAccInt version: ${LIBACCINT_VERSION_FULL}")
