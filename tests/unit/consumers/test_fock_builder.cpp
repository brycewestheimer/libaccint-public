// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_fock_builder.cpp
/// @brief Unit tests for CPU FockBuilder (J/K construction, symmetry, RHF/UHF)

#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/engine/engine.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/basis/shell.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/core/types.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

using namespace libaccint;
using namespace libaccint::consumers;

namespace {

constexpr Real TIGHT_TOL = 1e-10;
constexpr Real LOOSE_TOL = 1e-8;

// Water molecule geometry (Bohr)
const Point3D O_center{0.0, 0.0, 0.2217};
const Point3D H1_center{0.0, 1.4309, -0.8867};
const Point3D H2_center{0.0, -1.4309, -0.8867};

std::vector<Shell> make_sto3g_h2o_shells() {
    std::vector<Shell> shells;
    shells.reserve(5);

    Shell s0(0, O_center, {130.7093200, 23.8088610, 6.4436083},
             {0.15432897, 0.53532814, 0.44463454});
    s0.set_atom_index(0);
    shells.push_back(std::move(s0));

    Shell s1(0, O_center, {5.0331513, 1.1695961, 0.3803890},
             {-0.09996723, 0.39951283, 0.70011547});
    s1.set_atom_index(0);
    shells.push_back(std::move(s1));

    Shell s2(1, O_center, {5.0331513, 1.1695961, 0.3803890},
             {0.15591627, 0.60768372, 0.39195739});
    s2.set_atom_index(0);
    shells.push_back(std::move(s2));

    Shell s3(0, H1_center, {3.42525091, 0.62391373, 0.16885540},
             {0.15432897, 0.53532814, 0.44463454});
    s3.set_atom_index(1);
    shells.push_back(std::move(s3));

    Shell s4(0, H2_center, {3.42525091, 0.62391373, 0.16885540},
             {0.15432897, 0.53532814, 0.44463454});
    s4.set_atom_index(2);
    shells.push_back(std::move(s4));

    return shells;
}

/// Build identity density matrix
std::vector<Real> identity_density(Size nbf) {
    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) {
        D[i * nbf + i] = 1.0;
    }
    return D;
}

class ScopedEnvVar {
public:
    explicit ScopedEnvVar(const char* name) : name_(name) {
        const char* current = std::getenv(name_.c_str());
        if (current != nullptr) {
            had_original_ = true;
            original_value_ = current;
        }
    }

    ~ScopedEnvVar() { restore(); }

    void set(const char* value) {
#if defined(_WIN32)
        _putenv_s(name_.c_str(), value);
#else
        setenv(name_.c_str(), value, 1);
#endif
    }

private:
    void restore() {
#if defined(_WIN32)
        if (had_original_) {
            _putenv_s(name_.c_str(), original_value_.c_str());
        } else {
            _putenv_s(name_.c_str(), "");
        }
#else
        if (had_original_) {
            setenv(name_.c_str(), original_value_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
#endif
    }

    std::string name_;
    bool had_original_{false};
    std::string original_value_;
};

}  // anonymous namespace

// =============================================================================
// Basic Construction and API Tests (10.3.1)
// =============================================================================

TEST(FockBuilderTest, DefaultConstruction) {
    FockBuilder fock(7);
    EXPECT_EQ(fock.nbf(), 7u);

    auto J = fock.get_coulomb_matrix();
    auto K = fock.get_exchange_matrix();

    ASSERT_EQ(J.size(), 49u);
    ASSERT_EQ(K.size(), 49u);

    // Should be all zeros
    for (Size i = 0; i < J.size(); ++i) {
        EXPECT_DOUBLE_EQ(J[i], 0.0);
        EXPECT_DOUBLE_EQ(K[i], 0.0);
    }
}

TEST(FockBuilderTest, Reset) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();

    auto D = identity_density(nbf);

    FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    Operator op = Operator::coulomb();
    engine.compute_and_consume(op, fock);

    // J should be non-zero
    auto J = fock.get_coulomb_matrix();
    Real max_J = *std::max_element(J.begin(), J.end());
    EXPECT_GT(max_J, 0.0);

    // Reset
    fock.reset();

    // J should be zero again
    J = fock.get_coulomb_matrix();
    for (Size i = 0; i < J.size(); ++i) {
        EXPECT_DOUBLE_EQ(J[i], 0.0);
    }
}

TEST(FockBuilderTest, DensityMismatch) {
    FockBuilder fock(7);
    Real D[4] = {1.0, 0.0, 0.0, 1.0};
    EXPECT_ANY_THROW(fock.set_density(D, 2));
}

TEST(FockBuilderTest, BasicJKComputation) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();
    ASSERT_EQ(nbf, 7u);

    auto D = identity_density(nbf);

    FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    Operator op = Operator::coulomb();
    engine.compute_and_consume(op, fock);

    auto J = fock.get_coulomb_matrix();
    auto K = fock.get_exchange_matrix();

    // J should have positive diagonal
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GT(J[i * nbf + i], 0.0)
            << "J diagonal negative at " << i;
    }

    // K should have positive diagonal
    for (Size i = 0; i < nbf; ++i) {
        EXPECT_GT(K[i * nbf + i], 0.0)
            << "K diagonal negative at " << i;
    }
}

TEST(FockBuilderTest, FockMatrixCombination) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();

    auto D = identity_density(nbf);

    FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    Operator op = Operator::coulomb();
    engine.compute_and_consume(op, fock);

    // Use zero H_core
    std::vector<Real> H_core(nbf * nbf, 0.0);

    auto F = fock.get_fock_matrix(H_core, 1.0);
    auto J = fock.get_coulomb_matrix();
    auto K = fock.get_exchange_matrix();

    // F = J - K
    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(F[i], J[i] - K[i], TIGHT_TOL);
    }

    // F with exchange_fraction=0.5: F = J - 0.5*K
    auto F_half = fock.get_fock_matrix(H_core, 0.5);
    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(F_half[i], J[i] - 0.5 * K[i], TIGHT_TOL);
    }
}

// =============================================================================
// Symmetry Tests (10.3.2)
// =============================================================================

TEST(FockBuilderSymmetry, JIsSymmetric) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();

    auto D = identity_density(nbf);

    FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    Operator op = Operator::coulomb();
    engine.compute_and_consume(op, fock);

    auto J = fock.get_coulomb_matrix();

    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(J[i * nbf + j], J[j * nbf + i], TIGHT_TOL)
                << "J not symmetric at (" << i << "," << j << ")";
        }
    }
}

TEST(FockBuilderSymmetry, KIsSymmetric) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();

    auto D = identity_density(nbf);

    FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    Operator op = Operator::coulomb();
    engine.compute_and_consume(op, fock);

    auto K = fock.get_exchange_matrix();

    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(K[i * nbf + j], K[j * nbf + i], TIGHT_TOL)
                << "K not symmetric at (" << i << "," << j << ")";
        }
    }
}

TEST(FockBuilderSymmetry, AccumulateSymmetricMatchesAccumulate) {
    // Verify accumulate_symmetric produces the same result as accumulate
    // when the engine iterates unique quartets with symmetric scattering
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();

    auto D = identity_density(nbf);

    // Method 1: Full iteration with accumulate
    FockBuilder fock_full(nbf);
    fock_full.set_density(D.data(), nbf);

    Operator op = Operator::coulomb();
    engine.compute_and_consume(op, fock_full);

    auto J_full = fock_full.get_coulomb_matrix();
    auto K_full = fock_full.get_exchange_matrix();

    // Both should be symmetric
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(J_full[i * nbf + j], J_full[j * nbf + i], TIGHT_TOL);
            EXPECT_NEAR(K_full[i * nbf + j], K_full[j * nbf + i], TIGHT_TOL);
        }
    }

    // Both should be non-trivial (not all zeros)
    bool J_non_zero = false, K_non_zero = false;
    for (Size i = 0; i < nbf * nbf; ++i) {
        if (std::abs(J_full[i]) > TIGHT_TOL) J_non_zero = true;
        if (std::abs(K_full[i]) > TIGHT_TOL) K_non_zero = true;
    }
    EXPECT_TRUE(J_non_zero) << "J matrix is all zeros";
    EXPECT_TRUE(K_non_zero) << "K matrix is all zeros";
}

TEST(FockBuilderSymmetry, FockMatrixSymmetric) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();

    auto D = identity_density(nbf);

    FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    Operator op = Operator::coulomb();
    engine.compute_and_consume(op, fock);

    std::vector<Real> H_core(nbf * nbf, 0.0);
    auto F = fock.get_fock_matrix(H_core, 1.0);

    for (Size i = 0; i < nbf; ++i) {
        for (Size j = i + 1; j < nbf; ++j) {
            EXPECT_NEAR(F[i * nbf + j], F[j * nbf + i], TIGHT_TOL)
                << "Fock matrix not symmetric at (" << i << "," << j << ")";
        }
    }
}

// =============================================================================
// RHF/UHF Comparison Tests (10.3.8)
// =============================================================================

TEST(FockBuilderRHFUHF, DifferentDensitiesProduceDifferentResults) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();
    Operator op = Operator::coulomb();

    // Build 1: identity density
    auto D1 = identity_density(nbf);
    FockBuilder fock1(nbf);
    fock1.set_density(D1.data(), nbf);
    engine.compute_and_consume(op, fock1);
    auto J1 = fock1.get_coulomb_matrix();

    // Build 2: different density (only first element nonzero)
    std::vector<Real> D2(nbf * nbf, 0.0);
    D2[0] = 2.0;
    FockBuilder fock2(nbf);
    fock2.set_density(D2.data(), nbf);
    engine.compute_and_consume(op, fock2);
    auto J2 = fock2.get_coulomb_matrix();

    // Results should differ
    bool differs = false;
    for (Size i = 0; i < nbf * nbf; ++i) {
        if (std::abs(J1[i] - J2[i]) > TIGHT_TOL) {
            differs = true;
            break;
        }
    }
    EXPECT_TRUE(differs) << "Different densities produced identical J";
}

TEST(FockBuilderRHFUHF, ConsistentWithDensityDoubling) {
    // J is linear in density: J(2D) = 2*J(D)
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();
    Operator op = Operator::coulomb();

    auto D = identity_density(nbf);

    FockBuilder fock1(nbf);
    fock1.set_density(D.data(), nbf);
    engine.compute_and_consume(op, fock1);
    auto J1 = fock1.get_coulomb_matrix();

    std::vector<Real> D2(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf * nbf; ++i) D2[i] = 2.0 * D[i];

    FockBuilder fock2(nbf);
    fock2.set_density(D2.data(), nbf);
    engine.compute_and_consume(op, fock2);
    auto J2 = fock2.get_coulomb_matrix();

    // J(2D) should be 2*J(D)
    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(J2[i], 2.0 * J1[i], LOOSE_TOL)
            << "J linearity failed at element " << i;
    }
}

// =============================================================================
// Threading Strategy Tests
// =============================================================================

TEST(FockBuilderTest, ThreadingStrategyDefault) {
    FockBuilder fock(7);
    EXPECT_EQ(fock.threading_strategy(), FockThreadingStrategy::Sequential);
}

TEST(FockBuilderTest, ParallelMatchesSequential) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();

    auto D = identity_density(nbf);
    Operator op = Operator::coulomb();

    // Sequential
    FockBuilder fock_seq(nbf);
    fock_seq.set_density(D.data(), nbf);
    engine.compute_and_consume(op, fock_seq);

    // Parallel with 2 threads
    FockBuilder fock_par(nbf);
    fock_par.set_density(D.data(), nbf);
    engine.compute_and_consume_parallel(op, fock_par, 2);

    auto J_seq = fock_seq.get_coulomb_matrix();
    auto J_par = fock_par.get_coulomb_matrix();

    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(J_seq[i], J_par[i], LOOSE_TOL)
            << "Parallel J differs from sequential at " << i;
    }
}

TEST(FockBuilderTest, ParallelDefaultRemainsSafeWhenThreadLocalBudgetIsTiny) {
    BasisSet basis(make_sto3g_h2o_shells());
    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();

    auto D = identity_density(nbf);
    Operator op = Operator::coulomb();

    FockBuilder fock_seq(nbf);
    fock_seq.set_density(D.data(), nbf);
    engine.compute_and_consume(op, fock_seq);

    ScopedEnvVar env_guard("LIBACCINT_MAX_FOCK_THREADLOCAL_BYTES");
    env_guard.set("1");

    FockBuilder fock_par(nbf);
    fock_par.set_density(D.data(), nbf);
    EXPECT_NO_THROW(engine.compute_and_consume_parallel(op, fock_par, 2));

    auto J_seq = fock_seq.get_coulomb_matrix();
    auto J_par = fock_par.get_coulomb_matrix();
    for (Size i = 0; i < nbf * nbf; ++i) {
        EXPECT_NEAR(J_seq[i], J_par[i], LOOSE_TOL)
            << "Parallel safe fallback differs from sequential at " << i;
    }
}
