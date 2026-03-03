// test_version.cpp — Test version API
// Task 26.4.1: Verify version information is accessible and consistent
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include <libaccint/config.hpp>

#include <cstring>
#include <regex>
#include <string>

namespace {

// =============================================================================
// Version Macro Tests
// =============================================================================

TEST(VersionTest, MajorVersionDefined) {
    // LIBACCINT_VERSION_MAJOR should be a non-negative integer
    EXPECT_GE(LIBACCINT_VERSION_MAJOR, 0);
}

TEST(VersionTest, MinorVersionDefined) {
    EXPECT_GE(LIBACCINT_VERSION_MINOR, 0);
}

TEST(VersionTest, PatchVersionDefined) {
    EXPECT_GE(LIBACCINT_VERSION_PATCH, 0);
}

TEST(VersionTest, VersionStringDefined) {
    // LIBACCINT_VERSION should be a non-empty string
    EXPECT_NE(LIBACCINT_VERSION, nullptr);
    EXPECT_GT(std::strlen(LIBACCINT_VERSION), 0u);
}

TEST(VersionTest, VersionStringFormat) {
    // Version string should match "X.Y.Z" format
    std::string version(LIBACCINT_VERSION);
    std::regex semver_regex(R"(\d+\.\d+\.\d+.*)");
    EXPECT_TRUE(std::regex_match(version, semver_regex))
        << "Version string '" << version << "' does not match semver format";
}

TEST(VersionTest, VersionComponentsMatchString) {
    // Construct expected version string from components
    std::string expected = std::to_string(LIBACCINT_VERSION_MAJOR) + "." +
                           std::to_string(LIBACCINT_VERSION_MINOR) + "." +
                           std::to_string(LIBACCINT_VERSION_PATCH);
    std::string actual(LIBACCINT_VERSION);

    // The version string should start with the component-derived version
    EXPECT_EQ(actual.substr(0, expected.size()), expected)
        << "Version components (" << expected << ") don't match version string (" << actual << ")";
}

// =============================================================================
// Version Function Tests
// =============================================================================

TEST(VersionTest, VersionFunction) {
    const char* version = libaccint::version();
    EXPECT_NE(version, nullptr);
    EXPECT_GT(std::strlen(version), 0u);
    EXPECT_STREQ(version, LIBACCINT_VERSION);
}

TEST(VersionTest, VersionFunctionConstexpr) {
    // version() should be usable in constexpr context
    constexpr const char* version = libaccint::version();
    EXPECT_NE(version, nullptr);
}

// =============================================================================
// Backend Query Function Tests
// =============================================================================

TEST(VersionTest, BackendQueryFunctions) {
    // These should compile and return consistent values
    [[maybe_unused]] bool cuda = libaccint::has_cuda_backend();
    [[maybe_unused]] bool openmp = libaccint::has_openmp();
}

TEST(VersionTest, VectorizationInfo) {
    const char* isa = libaccint::vector_isa();
    EXPECT_NE(isa, nullptr);

    int width = libaccint::vector_width();
    EXPECT_GT(width, 0);
}

TEST(VersionTest, BackendConsistency) {
    // Verify macro and function values are consistent
    EXPECT_EQ(libaccint::has_cuda_backend(), static_cast<bool>(LIBACCINT_USE_CUDA));
    EXPECT_EQ(libaccint::has_openmp(), static_cast<bool>(LIBACCINT_USE_OPENMP));
}

// =============================================================================
// Compile-Time Version Checks
// =============================================================================

TEST(VersionTest, CurrentVersionIsV0) {
    // Verify we're building v0.x (pre-1.0 release)
    EXPECT_EQ(LIBACCINT_VERSION_MAJOR, 0);
}

} // namespace
