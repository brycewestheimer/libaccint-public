// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_examples.cpp
/// @brief Tests validating that example program patterns compile and work
///        correctly. These are simplified versions of the example programs
///        that verify the API usage patterns are correct.

#include <libaccint/libaccint.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/data/bse_json_parser.hpp>
#include <libaccint/config.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace libaccint;

namespace {

/// Helper: create H₂ atoms
std::vector<data::Atom> make_h2_atoms() {
    return {
        {1, {0.0, 0.0, 0.0}},
        {1, {0.0, 0.0, 1.4}},
    };
}

/// Helper: create water atoms
std::vector<data::Atom> make_water_atoms() {
    return {
        {8, {0.0000,  0.0000, 0.2217}},
        {1, {0.0000,  1.4309, -0.8867}},
        {1, {0.0000, -1.4309, -0.8867}},
    };
}

/// Helper: create nuclear PointChargeParams from atoms
PointChargeParams make_nuclear_charges(const std::vector<data::Atom>& atoms) {
    PointChargeParams charges;
    for (const auto& atom : atoms) {
        charges.x.push_back(atom.position.x);
        charges.y.push_back(atom.position.y);
        charges.z.push_back(atom.position.z);
        charges.charge.push_back(static_cast<Real>(atom.atomic_number));
    }
    return charges;
}

}  // anonymous namespace

// ============================================================================
// Task 28.1.1: Basic HF Energy Pattern
// ============================================================================

TEST(ExamplePatternTest, BasicHfEnergy) {
    auto atoms = make_h2_atoms();
    BasisSet basis = data::create_builtin_basis("STO-3G", atoms);
    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();

    EXPECT_EQ(nbf, 2u);

    // One-electron integrals
    std::vector<Real> S(nbf * nbf, 0.0);
    engine.compute_overlap_matrix(S);

    // S should be symmetric with diagonal = 1
    EXPECT_NEAR(S[0], 1.0, 1e-10);
    EXPECT_NEAR(S[3], 1.0, 1e-10);
    EXPECT_NEAR(S[1], S[2], 1e-14);

    // Core Hamiltonian
    auto charges = make_nuclear_charges(atoms);
    std::vector<Real> H_core(nbf * nbf, 0.0);
    engine.compute_core_hamiltonian(charges, H_core);

    // H_core diagonal should be negative (binding energy)
    EXPECT_LT(H_core[0], 0.0);
    EXPECT_LT(H_core[3], 0.0);
}

// ============================================================================
// Task 28.1.2: GPU Fock Build Pattern (CPU fallback)
// ============================================================================

TEST(ExamplePatternTest, GpuFockBuildCpuFallback) {
    auto atoms = make_water_atoms();
    BasisSet basis = data::create_builtin_basis("STO-3G", atoms);
    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();

    std::vector<Real> D(nbf * nbf, 0.0);
    for (Size i = 0; i < nbf; ++i) D[i * nbf + i] = 1.0;

    consumers::FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);

    // Force CPU backend
    engine.compute(Operator::coulomb(), fock, BackendHint::ForceCPU);

    auto J = fock.get_coulomb_matrix();
    // J should be symmetric and non-trivial
    EXPECT_NEAR(J[0 * nbf + 1], J[1 * nbf + 0], 1e-12);
}

// ============================================================================
// Task 28.1.3: QM/MM Embedding Pattern
// ============================================================================

TEST(ExamplePatternTest, QmmEmbedding) {
    auto qm_atoms = make_water_atoms();
    BasisSet basis = data::create_builtin_basis("STO-3G", qm_atoms);
    Engine engine(basis);
    const Size nbf = basis.n_basis_functions();

    // QM core Hamiltonian
    auto nuclear = make_nuclear_charges(qm_atoms);
    std::vector<Real> H_qm(nbf * nbf, 0.0);
    engine.compute_core_hamiltonian(nuclear, H_qm);

    // MM point charges
    PointChargeParams mm_charges;
    mm_charges.x = {6.0};
    mm_charges.y = {0.0};
    mm_charges.z = {0.0};
    mm_charges.charge = {-0.834};

    std::vector<Real> V_mm(nbf * nbf, 0.0);
    auto op_mm = OneElectronOperator(Operator::point_charges(mm_charges));
    engine.compute(op_mm, V_mm);

    // MM contribution should be nonzero
    Real sum_mm = 0.0;
    for (const auto& v : V_mm) sum_mm += std::abs(v);
    EXPECT_GT(sum_mm, 0.0);
}

// ============================================================================
// Task 28.1.5: Density Fitting Pattern
// ============================================================================

TEST(ExamplePatternTest, DFBasisSetConstruction) {
    auto atoms = make_h2_atoms();
    BasisSet orbital_basis = data::create_builtin_basis("STO-3G", atoms);

    // Create auxiliary shells
    std::vector<Shell> aux_shells;
    for (const auto& atom : atoms) {
        aux_shells.emplace_back(0, atom.position,
                                std::vector<Real>{1.0, 0.3},
                                std::vector<Real>{0.5, 0.5});
    }

    AuxiliaryBasisSet aux_basis(std::move(aux_shells));

    EXPECT_GT(aux_basis.n_shells(), 0u);
    EXPECT_GT(aux_basis.n_functions(), 0u);
}

// ============================================================================
// Task 28.1.6: Custom Consumer Pattern
// ============================================================================

namespace {
/// Test consumer that counts integrals
struct TestCounter {
    Size count{0};
    void accumulate(const TwoElectronBuffer<0>& buffer,
                    [[maybe_unused]] Index fa, [[maybe_unused]] Index fb,
                    [[maybe_unused]] Index fc, [[maybe_unused]] Index fd,
                    int na, int nb, int nc, int nd) {
        count += static_cast<Size>(na * nb * nc * nd);
        (void)buffer;
    }
    void prepare_parallel([[maybe_unused]] int n_threads) {}
    void finalize_parallel() {}
};
}  // anonymous namespace

TEST(ExamplePatternTest, CustomConsumer) {
    auto atoms = make_h2_atoms();
    BasisSet basis = data::create_builtin_basis("STO-3G", atoms);
    Engine engine(basis);

    TestCounter counter;
    engine.compute(Operator::coulomb(), counter);

    // H₂/STO-3G: 2x2x2x2 = 16 integrals total from unique shell quartets
    EXPECT_GT(counter.count, 0u);
}

// ============================================================================
// Task 28.2.1: BSE JSON Parser
// ============================================================================

TEST(ExamplePatternTest, BseJsonParserBasic) {
    const std::string json = R"({
        "molssi_bse_schema": {"schema_type": "complete", "schema_version": "0.1"},
        "name": "Test",
        "elements": {
            "1": {
                "electron_shells": [{
                    "function_type": "gto",
                    "angular_momentum": [0],
                    "exponents": ["3.42525091", "0.62391373", "0.16885540"],
                    "coefficients": [["0.15432897", "0.53532814", "0.44463454"]]
                }]
            }
        }
    })";

    std::vector<data::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
    BasisSet basis = data::BseJsonParser::parse(json, atoms);
    EXPECT_EQ(basis.n_shells(), 1u);
    EXPECT_EQ(basis.n_basis_functions(), 1u);
}

// ============================================================================
// Version API
// ============================================================================

TEST(ExamplePatternTest, VersionApi) {
    const char* ver = version();
    EXPECT_NE(ver, nullptr);
    EXPECT_GT(std::string(ver).length(), 0u);
}
