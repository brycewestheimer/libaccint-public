// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file spherical_transform_2e.cpp
/// @brief Implementation of two-electron spherical transformation

#include "spherical_transform_2e.hpp"
#include <libaccint/math/spherical_transform.hpp>

namespace libaccint::transforms {

void transform_2e_block(int La, int Lb, int Lc, int Ld,
                        const double* cartesian, double* spherical,
                        double* work) {
    math::transform_4d(La, Lb, Lc, Ld, cartesian, spherical, work);
}

// =============================================================================
// SphericalTransform2E Class Implementation
// =============================================================================

SphericalTransform2E::SphericalTransform2E(int max_am)
    : max_am_(max_am),
      transformer_(max_am) {
    // Pre-allocate work buffer for worst-case quartet
    work_.resize(math::work_size_4d(max_am, max_am, max_am, max_am));
}

void SphericalTransform2E::transform(int La, int Lb, int Lc, int Ld,
                                      const double* cartesian, double* spherical) {
    transform_2e_block(La, Lb, Lc, Ld, cartesian, spherical, work_.data());
}

// =============================================================================
// Full Tensor Transformation
// =============================================================================

void transform_2e_tensor(const BasisSet& basis,
                         const std::vector<double>& cartesian_tensor,
                         std::vector<double>& spherical_tensor) {
    const Size n_sph = basis.n_basis_functions_spherical();

    // Resize output
    spherical_tensor.resize(n_sph * n_sph * n_sph * n_sph, 0.0);

    // Create transformer
    const int max_am = basis.max_angular_momentum();
    SphericalTransform2E transformer(max_am);

    // Compute spherical offsets for each shell
    std::vector<Index> sph_offsets(basis.n_shells());
    Index sph_offset = 0;
    for (Size i = 0; i < basis.n_shells(); ++i) {
        sph_offsets[i] = sph_offset;
        sph_offset += n_spherical(basis.shell(i).angular_momentum());
    }

    const Size n_cart = basis.n_basis_functions();

    // Pre-allocate work buffers once, sized for worst-case quartet
    const int max_n_cart = n_cartesian(max_am);
    const int max_n_sph = n_spherical(max_am);
    std::vector<double> cart_block(max_n_cart * max_n_cart * max_n_cart * max_n_cart);
    std::vector<double> sph_block(max_n_sph * max_n_sph * max_n_sph * max_n_sph);

    // Transform each shell quartet
    for (Size a = 0; a < basis.n_shells(); ++a) {
        const Shell& shell_a = basis.shell(a);
        const int La = shell_a.angular_momentum();
        const int na_cart = n_cartesian(La), na_sph = n_spherical(La);
        const Index cart_off_a = shell_a.function_index();
        const Index sph_off_a = sph_offsets[a];

        for (Size b = 0; b < basis.n_shells(); ++b) {
            const Shell& shell_b = basis.shell(b);
            const int Lb = shell_b.angular_momentum();
            const int nb_cart = n_cartesian(Lb), nb_sph = n_spherical(Lb);
            const Index cart_off_b = shell_b.function_index();
            const Index sph_off_b = sph_offsets[b];

            for (Size c = 0; c < basis.n_shells(); ++c) {
                const Shell& shell_c = basis.shell(c);
                const int Lc = shell_c.angular_momentum();
                const int nc_cart = n_cartesian(Lc), nc_sph = n_spherical(Lc);
                const Index cart_off_c = shell_c.function_index();
                const Index sph_off_c = sph_offsets[c];

                for (Size d = 0; d < basis.n_shells(); ++d) {
                    const Shell& shell_d = basis.shell(d);
                    const int Ld = shell_d.angular_momentum();
                    const int nd_cart = n_cartesian(Ld), nd_sph = n_spherical(Ld);
                    const Index cart_off_d = shell_d.function_index();
                    const Index sph_off_d = sph_offsets[d];

                    // Extract Cartesian block (reuse pre-allocated buffer)
                    for (int ia = 0; ia < na_cart; ++ia) {
                        for (int ib = 0; ib < nb_cart; ++ib) {
                            for (int ic = 0; ic < nc_cart; ++ic) {
                                for (int id = 0; id < nd_cart; ++id) {
                                    Index cart_idx =
                                        (((cart_off_a + ia) * n_cart + (cart_off_b + ib)) * n_cart +
                                         (cart_off_c + ic)) * n_cart + (cart_off_d + id);
                                    cart_block[((ia * nb_cart + ib) * nc_cart + ic) * nd_cart + id] =
                                        cartesian_tensor[cart_idx];
                                }
                            }
                        }
                    }

                    // Transform to spherical (reuse pre-allocated buffer)
                    transformer.transform(La, Lb, Lc, Ld,
                                          cart_block.data(), sph_block.data());

                    // Store spherical block
                    for (int ia = 0; ia < na_sph; ++ia) {
                        for (int ib = 0; ib < nb_sph; ++ib) {
                            for (int ic = 0; ic < nc_sph; ++ic) {
                                for (int id = 0; id < nd_sph; ++id) {
                                    Index sph_idx =
                                        (((sph_off_a + ia) * n_sph + (sph_off_b + ib)) * n_sph +
                                         (sph_off_c + ic)) * n_sph + (sph_off_d + id);
                                    spherical_tensor[sph_idx] =
                                        sph_block[((ia * nb_sph + ib) * nc_sph + ic) * nd_sph + id];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace libaccint::transforms
