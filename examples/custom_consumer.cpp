// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.
//
// custom_consumer.cpp
//
// Demonstrates implementing a custom IntegralConsumer for the compute-and-
// consume pattern. Custom consumers allow users to process integrals on-the-fly
// without materializing the full integral tensor.
//
// This example implements:
//   1. A simple consumer that counts non-zero integrals
//   2. A consumer that computes the trace of the ERI tensor times density
//
// Key concepts:
//   - The accumulate() interface
//   - Shell quartet indexing (function offsets and counts)
//   - Integration with Engine::compute() and Engine::compute_and_consume()

#include <libaccint/libaccint.hpp>
#include <libaccint/data/builtin_basis.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace libaccint;

// ============================================================================
// Custom Consumer 1: Integral Counter
// ============================================================================

/// @brief A simple consumer that counts significant integrals
///
/// This consumer processes each shell quartet's integral buffer and counts
/// the number of integrals exceeding a given threshold. This is useful for
/// analyzing integral sparsity and estimating screening efficiency.
class IntegralCounter {
public:
    /// @brief Construct a counter with a significance threshold
    explicit IntegralCounter(Real threshold = 1e-12)
        : threshold_(threshold) {}

    /// @brief Accumulate method required by Engine::compute()
    ///
    /// The Engine calls this for each shell quartet (a b | c d).
    /// Parameters provide the function indices and counts for mapping
    /// buffer elements to basis function indices.
    void accumulate(const TwoElectronBuffer<0>& buffer,
                    Index fa, Index fb, Index fc, Index fd,
                    int na, int nb, int nc, int nd) {
        total_quartets_++;

        for (int a = 0; a < na; ++a) {
            for (int b = 0; b < nb; ++b) {
                for (int c = 0; c < nc; ++c) {
                    for (int d = 0; d < nd; ++d) {
                        Real val = buffer(a, b, c, d);
                        total_integrals_++;
                        if (std::abs(val) > threshold_) {
                            significant_integrals_++;
                        }
                    }
                }
            }
        }

        // Suppress unused variable warnings
        (void)fa; (void)fb; (void)fc; (void)fd;
    }

    /// @brief Get total number of shell quartets processed
    [[nodiscard]] Size total_quartets() const noexcept { return total_quartets_; }

    /// @brief Get total number of integrals processed
    [[nodiscard]] Size total_integrals() const noexcept { return total_integrals_; }

    /// @brief Get number of significant integrals
    [[nodiscard]] Size significant_integrals() const noexcept { return significant_integrals_; }

    /// @brief Get sparsity (fraction of insignificant integrals)
    [[nodiscard]] double sparsity() const noexcept {
        if (total_integrals_ == 0) return 0.0;
        return 1.0 - static_cast<double>(significant_integrals_) /
                      static_cast<double>(total_integrals_);
    }

    void prepare_parallel([[maybe_unused]] int n_threads) {}
    void finalize_parallel() {}

private:
    Real threshold_;
    Size total_quartets_{0};
    Size total_integrals_{0};
    Size significant_integrals_{0};
};

// ============================================================================
// Custom Consumer 2: Coulomb Energy Calculator
// ============================================================================

/// @brief Consumer that computes E_J = 0.5 * sum_{μνλσ} D_μν * (μν|λσ) * D_λσ
///
/// Instead of building the full J matrix, this consumer directly accumulates
/// the Coulomb energy contribution. This avoids O(N²) storage for J.
class CoulombEnergyConsumer {
public:
    /// @brief Construct with density matrix
    /// @param D Pointer to density matrix (row-major, nbf x nbf)
    /// @param nbf Number of basis functions
    CoulombEnergyConsumer(const Real* D, Size nbf)
        : D_(D), nbf_(nbf) {}

    /// @brief Accumulate Coulomb energy contributions from a shell quartet
    void accumulate(const TwoElectronBuffer<0>& buffer,
                    Index fa, Index fb, Index fc, Index fd,
                    int na, int nb, int nc, int nd) {
        for (int a = 0; a < na; ++a) {
            for (int b = 0; b < nb; ++b) {
                Real D_ab = D_[static_cast<Size>(fa + a) * nbf_ +
                               static_cast<Size>(fb + b)];
                for (int c = 0; c < nc; ++c) {
                    for (int d = 0; d < nd; ++d) {
                        Real D_cd = D_[static_cast<Size>(fc + c) * nbf_ +
                                       static_cast<Size>(fd + d)];
                        Real eri = buffer(a, b, c, d);
                        energy_ += D_ab * eri * D_cd;
                    }
                }
            }
        }
    }

    /// @brief Get the accumulated Coulomb energy (0.5 * sum D*eri*D)
    [[nodiscard]] Real energy() const noexcept { return 0.5 * energy_; }

    void prepare_parallel([[maybe_unused]] int n_threads) {}
    void finalize_parallel() {}

private:
    const Real* D_;
    Size nbf_;
    Real energy_{0.0};
};

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== LibAccInt Custom Consumer Example ===\n";
    std::cout << "LibAccInt version: " << version() << "\n\n";

    // ── Set up H₂ / STO-3G ──────────────────────────────────────────────────
    std::vector<data::Atom> atoms = {
        {1, {0.0, 0.0, 0.0}},
        {1, {0.0, 0.0, 1.4}},
    };

    BasisSet basis = data::create_builtin_basis("STO-3G", atoms);
    const Size nbf = basis.n_basis_functions();
    Engine engine(basis);

    std::cout << "Molecule: H2 / STO-3G (" << nbf << " basis functions)\n\n";

    // ── Simple density matrix ───────────────────────────────────────────────
    std::vector<Real> D(nbf * nbf, 0.0);
    const Real c = 1.0 / std::sqrt(2.0);
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            D[i * nbf + j] = 2.0 * c * c;
        }
    }

    // ── Consumer 1: Integral Counter ────────────────────────────────────────
    IntegralCounter counter(1e-12);
    engine.compute(Operator::coulomb(), counter);

    std::cout << "=== Integral Counter Results ===\n";
    std::cout << "Shell quartets processed: " << counter.total_quartets() << "\n";
    std::cout << "Total integrals: " << counter.total_integrals() << "\n";
    std::cout << "Significant integrals (>1e-12): "
              << counter.significant_integrals() << "\n";
    std::cout << "Sparsity: " << std::fixed << std::setprecision(1)
              << (counter.sparsity() * 100.0) << "%\n\n";

    // ── Consumer 2: Coulomb Energy Calculator ───────────────────────────────
    CoulombEnergyConsumer energy_consumer(D.data(), nbf);
    engine.compute(Operator::coulomb(), energy_consumer);

    std::cout << "=== Coulomb Energy Consumer Results ===\n";
    std::cout << "E_J = " << std::fixed << std::setprecision(10)
              << energy_consumer.energy() << " Hartree\n\n";

    // ── Compare with FockBuilder ────────────────────────────────────────────
    consumers::FockBuilder fock(nbf);
    fock.set_density(D.data(), nbf);
    engine.compute(Operator::coulomb(), fock);

    // Compute E_J from FockBuilder: E_J = 0.5 * Tr[D * J]
    auto J = fock.get_coulomb_matrix();
    Real E_J_fock = 0.0;
    for (Size i = 0; i < nbf; ++i) {
        for (Size j = 0; j < nbf; ++j) {
            E_J_fock += D[i * nbf + j] * J[i * nbf + j];
        }
    }
    E_J_fock *= 0.5;

    std::cout << "E_J (FockBuilder): " << std::setprecision(10) << E_J_fock << " Hartree\n";
    std::cout << "Difference: " << std::scientific << std::setprecision(2)
              << std::abs(energy_consumer.energy() - E_J_fock) << "\n";

    return 0;
}
