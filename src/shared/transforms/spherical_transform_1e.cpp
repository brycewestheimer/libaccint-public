// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file spherical_transform_1e.cpp
/// @brief Implementation of one-electron spherical transformation

#include "spherical_transform_1e.hpp"
#include <libaccint/math/spherical_transform.hpp>

namespace libaccint::transforms {

void transform_1e_block(int La, int Lb,
                        const double* cartesian, double* spherical,
                        double* work) {
    math::transform_2d(La, Lb, cartesian, spherical, work);
}

void transform_1e_to_spherical(const BasisSet& basis,
                               const std::vector<double>& cartesian,
                               std::vector<double>& spherical) {
    const Size n_cart = basis.n_basis_functions();
    const Size n_sph = basis.n_basis_functions_spherical();

    // Resize output if needed
    spherical.resize(n_sph * n_sph, 0.0);

    // Work buffer for largest shell pair
    const int max_am = basis.max_angular_momentum();
    std::vector<double> work(math::work_size_2d(max_am, max_am));

    // Track spherical offsets for each shell
    std::vector<Index> sph_offsets(basis.n_shells());
    Index sph_offset = 0;
    for (Size i = 0; i < basis.n_shells(); ++i) {
        sph_offsets[i] = sph_offset;
        sph_offset += n_spherical(basis.shell(i).angular_momentum());
    }

    // Transform each shell pair block
    for (Size i = 0; i < basis.n_shells(); ++i) {
        const Shell& shell_i = basis.shell(i);
        const int La = shell_i.angular_momentum();
        const int n_cart_a = n_cartesian(La);
        const int n_sph_a = n_spherical(La);
        const Index cart_off_a = shell_i.function_index();
        const Index sph_off_a = sph_offsets[i];

        for (Size j = 0; j < basis.n_shells(); ++j) {
            const Shell& shell_j = basis.shell(j);
            const int Lb = shell_j.angular_momentum();
            const int n_cart_b = n_cartesian(Lb);
            const int n_sph_b = n_spherical(Lb);
            const Index cart_off_b = shell_j.function_index();
            const Index sph_off_b = sph_offsets[j];

            // Extract Cartesian block
            std::vector<double> cart_block(n_cart_a * n_cart_b);
            for (int a = 0; a < n_cart_a; ++a) {
                for (int b = 0; b < n_cart_b; ++b) {
                    cart_block[a * n_cart_b + b] =
                        cartesian[(cart_off_a + a) * n_cart + (cart_off_b + b)];
                }
            }

            // Transform to spherical
            std::vector<double> sph_block(n_sph_a * n_sph_b);
            transform_1e_block(La, Lb, cart_block.data(), sph_block.data(), work.data());

            // Store spherical block
            for (int a = 0; a < n_sph_a; ++a) {
                for (int b = 0; b < n_sph_b; ++b) {
                    spherical[(sph_off_a + a) * n_sph + (sph_off_b + b)] =
                        sph_block[a * n_sph_b + b];
                }
            }
        }
    }
}

// =============================================================================
// SphericalTransform1E Class Implementation
// =============================================================================

SphericalTransform1E::SphericalTransform1E(const BasisSet& basis)
    : basis_(&basis),
      n_cart_(basis.n_basis_functions()),
      n_sph_(basis.n_basis_functions_spherical()),
      transformer_(basis.max_angular_momentum()) {
    // Pre-allocate work buffer
    const int max_am = basis.max_angular_momentum();
    work_.resize(math::work_size_2d(max_am, max_am));
}

void SphericalTransform1E::transform(const std::vector<double>& cartesian,
                                      std::vector<double>& spherical) {
    transform_1e_to_spherical(*basis_, cartesian, spherical);
}

} // namespace libaccint::transforms
