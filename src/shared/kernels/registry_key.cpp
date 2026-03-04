// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/kernels/registry_key.hpp>

namespace libaccint::kernels {

RegistryKey RegistryKey::for_1e(OperatorKind kind, int la, int lb,
                                int na_prim, int nb_prim,
                                BackendType backend) {
    return RegistryKey{
        kind,
        {la, lb, 0, 0},
        {na_prim, nb_prim, 1, 1},  // Use 1 for unused centers to avoid zero product
        backend
    };
}

RegistryKey RegistryKey::for_2e(OperatorKind kind,
                                int la, int lb, int lc, int ld,
                                int na, int nb, int nc, int nd,
                                BackendType backend) {
    return RegistryKey{
        kind,
        {la, lb, lc, ld},
        {na, nb, nc, nd},
        backend
    };
}

std::size_t RegistryKey::Hash::operator()(const RegistryKey& k) const noexcept {
    // Combine hash values using FNV-1a-like mixing
    std::size_t h = 14695981039346656037ULL;  // FNV offset basis

    auto mix = [&h](std::size_t value) {
        h ^= value;
        h *= 1099511628211ULL;  // FNV prime
    };

    // Hash operator kind
    mix(static_cast<std::size_t>(k.op_kind));

    // Hash angular momentum quartet
    for (int am_val : k.am) {
        mix(static_cast<std::size_t>(am_val));
    }

    // Hash primitive counts
    for (int prim : k.n_primitives) {
        mix(static_cast<std::size_t>(prim));
    }

    // Hash backend type
    mix(static_cast<std::size_t>(k.available_backend));

    return h;
}

}  // namespace libaccint::kernels
