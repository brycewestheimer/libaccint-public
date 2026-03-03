// test_install_integration.cpp — Smoke test for installed library
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include <libaccint/config.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/core/backend.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/operators/operator_types.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <cstring>
#include <string>

namespace {

TEST(InstallIntegrationTest, ConfigHeaderAvailable) {
    EXPECT_GT(std::strlen(LIBACCINT_VERSION), 0u);
    EXPECT_GE(LIBACCINT_VERSION_MAJOR, 0);
}

TEST(InstallIntegrationTest, VersionFunctionAvailable) {
    const char* version = libaccint::version();
    EXPECT_NE(version, nullptr);
    EXPECT_STREQ(version, LIBACCINT_VERSION);
}

TEST(InstallIntegrationTest, CoreTypesUsable) {
    [[maybe_unused]] auto overlap = libaccint::OperatorKind::Overlap;
    [[maybe_unused]] auto kinetic = libaccint::OperatorKind::Kinetic;
    [[maybe_unused]] auto nuclear = libaccint::OperatorKind::Nuclear;
    [[maybe_unused]] auto coulomb = libaccint::OperatorKind::Coulomb;
}

TEST(InstallIntegrationTest, ShellCreation) {
    libaccint::Point3D origin{0.0, 0.0, 0.0};
    libaccint::Shell shell(0, origin, {1.0}, {1.0});
    EXPECT_EQ(shell.angular_momentum(), 0);
    EXPECT_EQ(shell.n_primitives(), 1u);
    EXPECT_EQ(shell.n_functions(), 1);
}

TEST(InstallIntegrationTest, BuiltinBasisAvailable) {
    try {
        libaccint::data::Atom h_atom{1, {0.0, 0.0, 0.0}};
        std::vector<libaccint::data::Atom> atoms = {h_atom};
        auto basis = libaccint::data::create_builtin_basis("STO-3G", atoms);
        EXPECT_GT(basis.n_shells(), 0u);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "STO-3G not available: " << e.what();
    }
}

TEST(InstallIntegrationTest, EngineCreation) {
    libaccint::Point3D origin{0.0, 0.0, 0.0};
    libaccint::Shell s_shell(0, origin, {3.42525091}, {0.15432897});
    std::vector<libaccint::Shell> shells = {s_shell};
    libaccint::BasisSet basis(shells);
    [[maybe_unused]] libaccint::Engine engine(basis);
}

TEST(InstallIntegrationTest, VersionIsSemanticVersion) {
    std::string version(LIBACCINT_VERSION);
    auto dot_count = std::count(version.begin(), version.end(), '.');
    EXPECT_GE(dot_count, 2) << "Version is not semver: " << version;
}

} // namespace
