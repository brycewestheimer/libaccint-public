// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/utils/matrix_assembly.hpp>
#include <libaccint/utils/error_handling.hpp>

namespace libaccint {

void assemble_one_electron_matrix(Engine& engine,
                                  const OneElectronOperator& op,
                                  std::span<Real> matrix) {
    const Size nbf = engine.basis().n_basis_functions();
    if (matrix.size() != nbf * nbf) {
        throw InvalidArgumentException(
            "Matrix size must be nbf * nbf = " + std::to_string(nbf * nbf) +
            ", got " + std::to_string(matrix.size()));
    }

    // Use Engine's compute_1e which handles all the assembly
    std::vector<Real> result;
    engine.compute_1e(op, result);

    // Copy to output span
    std::copy(result.begin(), result.end(), matrix.begin());
}

std::vector<Real> assemble_one_electron_matrix(Engine& engine,
                                               const OneElectronOperator& op) {
    std::vector<Real> result;
    engine.compute_1e(op, result);
    return result;
}

}  // namespace libaccint
