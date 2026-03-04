// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

#include <libaccint/basis/shell_set_pair.hpp>

namespace libaccint {

/**
 * @brief Quartet of ShellSets for two-electron integrals.
 *
 * ShellSetQuartet represents the combination of two ShellSetPairs (bra and ket),
 * used for computing two-electron integrals of the form (ab|cd).
 *
 * This is a lightweight class containing two pointers and cached metadata.
 * For MVP (Phase 1), Schwarz screening is not implemented and returns 1.0
 * (the product of bra and ket Schwarz bounds).
 */
class ShellSetQuartet {
public:
    /**
     * @brief Constructs a ShellSetQuartet from bra and ket ShellSetPairs.
     * @param bra Bra ShellSetPair (ab)
     * @param ket Ket ShellSetPair (cd)
     */
    ShellSetQuartet(const ShellSetPair& bra, const ShellSetPair& ket) noexcept
        : bra_(&bra), ket_(&ket) {}

    /**
     * @brief Returns a reference to the bra ShellSetPair.
     */
    [[nodiscard]] const ShellSetPair& bra_pair() const noexcept { return *bra_; }

    /**
     * @brief Returns a reference to the ket ShellSetPair.
     */
    [[nodiscard]] const ShellSetPair& ket_pair() const noexcept { return *ket_; }

    /**
     * @brief Returns the angular momentum of center A (bra, first shell).
     */
    [[nodiscard]] int La() const noexcept { return bra_->La(); }

    /**
     * @brief Returns the angular momentum of center B (bra, second shell).
     */
    [[nodiscard]] int Lb() const noexcept { return bra_->Lb(); }

    /**
     * @brief Returns the angular momentum of center C (ket, first shell).
     */
    [[nodiscard]] int Lc() const noexcept { return ket_->La(); }

    /**
     * @brief Returns the angular momentum of center D (ket, second shell).
     */
    [[nodiscard]] int Ld() const noexcept { return ket_->Lb(); }

    /**
     * @brief Returns the total angular momentum (La + Lb + Lc + Ld).
     */
    [[nodiscard]] int L_total() const noexcept {
        return La() + Lb() + Lc() + Ld();
    }

    /**
     * @brief Returns the total number of shell quartets.
     * @return n_pairs_bra * n_pairs_ket
     */
    [[nodiscard]] Size n_quartets() const noexcept {
        return bra_->n_pairs() * ket_->n_pairs();
    }

    /**
     * @brief Returns Schwarz screening bound for this quartet.
     * @return Product of bra and ket Schwarz bounds (1.0 for MVP)
     */
    [[nodiscard]] Real schwarz_bound() const noexcept {
        return bra_->schwarz_bound() * ket_->schwarz_bound();
    }

private:
    const ShellSetPair* bra_;  ///< Pointer to bra ShellSetPair (ab)
    const ShellSetPair* ket_;  ///< Pointer to ket ShellSetPair (cd)
};

}  // namespace libaccint
