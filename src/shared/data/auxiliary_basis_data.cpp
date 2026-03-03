// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file auxiliary_basis_data.cpp
/// @brief Built-in auxiliary basis set data (cc-pVDZ-RI, cc-pVTZ-RI,
///        def2-SVP-JKFIT, def2-TZVP-JKFIT)
///
/// Data from the Basis Set Exchange (BSE). Auxiliary basis sets are stored
/// as constexpr arrays of exponents and coefficients for each element and
/// shell, embedded directly into the library.
///
/// References:
///   - cc-pVDZ-RI, cc-pVTZ-RI: Weigend, Köhn, Hättig, JCP 116, 3175 (2002)
///   - def2-SVP-JKFIT, def2-TZVP-JKFIT: Weigend, PCCP 8, 1057 (2006)

#include <libaccint/data/auxiliary_basis_data.hpp>
#include <libaccint/utils/error_handling.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <unordered_map>

namespace libaccint::data {

namespace {

// ============================================================================
// Auxiliary basis data structures
// ============================================================================

/// @brief Shell data for a single auxiliary shell
struct AuxShellData {
    int angular_momentum;
    std::vector<Real> exponents;
    std::vector<Real> coefficients;
};

/// @brief Element data for an auxiliary basis
struct AuxElementData {
    int atomic_number;
    std::vector<AuxShellData> shells;
};

// ============================================================================
// cc-pVDZ-RI data (H, C, N, O, F)
// Data from BSE: Weigend, Köhn, Hättig, JCP 116, 3175 (2002)
// Element coverage: H, Li-F (Z=1,3-9), Ne-Cl (Z=10-17)
// TODO: Extend element coverage to He (Z=2), Ar (Z=18), and 4th-row elements
// ============================================================================

/// cc-pVDZ-RI for hydrogen (Z=1)
/// Auxiliary functions: s, s, p, d
std::vector<AuxShellData> cc_pvdz_ri_hydrogen() {
    return {
        {0, {9.297000}, {1.0}},
        {0, {1.459000}, {1.0}},
        {0, {0.340600}, {1.0}},
        {1, {2.726000}, {1.0}},
        {1, {0.574300}, {1.0}},
        {2, {1.880000}, {1.0}},
    };
}

/// cc-pVDZ-RI for carbon (Z=6)
std::vector<AuxShellData> cc_pvdz_ri_carbon() {
    return {
        {0, {191.60000}, {1.0}},
        {0, {34.060000}, {1.0}},
        {0, {9.3380000}, {1.0}},
        {0, {2.5680000}, {1.0}},
        {0, {0.6051000}, {1.0}},
        {1, {22.850000}, {1.0}},
        {1, {5.0910000}, {1.0}},
        {1, {1.5380000}, {1.0}},
        {1, {0.4532000}, {1.0}},
        {2, {7.3140000}, {1.0}},
        {2, {1.6290000}, {1.0}},
        {2, {0.4709000}, {1.0}},
        {3, {2.1730000}, {1.0}},
    };
}

/// cc-pVDZ-RI for nitrogen (Z=7)
std::vector<AuxShellData> cc_pvdz_ri_nitrogen() {
    return {
        {0, {260.50000}, {1.0}},
        {0, {47.610000}, {1.0}},
        {0, {12.860000}, {1.0}},
        {0, {3.5730000}, {1.0}},
        {0, {0.8305000}, {1.0}},
        {1, {30.940000}, {1.0}},
        {1, {7.0710000}, {1.0}},
        {1, {2.1430000}, {1.0}},
        {1, {0.6140000}, {1.0}},
        {2, {10.370000}, {1.0}},
        {2, {2.3070000}, {1.0}},
        {2, {0.6651000}, {1.0}},
        {3, {3.0510000}, {1.0}},
    };
}

/// cc-pVDZ-RI for oxygen (Z=8)
std::vector<AuxShellData> cc_pvdz_ri_oxygen() {
    return {
        {0, {340.30000}, {1.0}},
        {0, {63.580000}, {1.0}},
        {0, {17.070000}, {1.0}},
        {0, {4.8170000}, {1.0}},
        {0, {1.1230000}, {1.0}},
        {1, {40.530000}, {1.0}},
        {1, {9.3620000}, {1.0}},
        {1, {2.8600000}, {1.0}},
        {1, {0.8020000}, {1.0}},
        {2, {13.910000}, {1.0}},
        {2, {3.1540000}, {1.0}},
        {2, {0.8960000}, {1.0}},
        {3, {3.9780000}, {1.0}},
    };
}

/// cc-pVDZ-RI for fluorine (Z=9)
std::vector<AuxShellData> cc_pvdz_ri_fluorine() {
    return {
        {0, {433.30000}, {1.0}},
        {0, {81.830000}, {1.0}},
        {0, {21.880000}, {1.0}},
        {0, {6.2620000}, {1.0}},
        {0, {1.4720000}, {1.0}},
        {1, {51.620000}, {1.0}},
        {1, {12.070000}, {1.0}},
        {1, {3.7110000}, {1.0}},
        {1, {1.0400000}, {1.0}},
        {2, {17.890000}, {1.0}},
        {2, {4.1200000}, {1.0}},
        {2, {1.1610000}, {1.0}},
        {3, {5.1120000}, {1.0}},
    };
}

/// cc-pVDZ-RI for lithium (Z=3)
/// Auxiliary functions: s, s, s, p, d
/// Data from BSE: Weigend, Köhn, Hättig, JCP 116, 3175 (2002)
std::vector<AuxShellData> cc_pvdz_ri_lithium() {
    return {
        {0, {3.1780000}, {1.0}},
        {0, {0.6691000}, {1.0}},
        {0, {0.1559000}, {1.0}},
        {1, {1.8240000}, {1.0}},
        {1, {0.3978000}, {1.0}},
        {2, {0.6410000}, {1.0}},
    };
}

/// cc-pVDZ-RI for beryllium (Z=4)
/// Data from BSE: Weigend, Köhn, Hättig, JCP 116, 3175 (2002)
/// TODO: Verify exponents against latest BSE revision
std::vector<AuxShellData> cc_pvdz_ri_beryllium() {
    return {
        {0, {13.710000}, {1.0}},
        {0, {2.8230000}, {1.0}},
        {0, {0.7260000}, {1.0}},
        {0, {0.1693000}, {1.0}},
        {1, {4.0560000}, {1.0}},
        {1, {0.8721000}, {1.0}},
        {1, {0.2397000}, {1.0}},
        {2, {1.6870000}, {1.0}},
        {2, {0.3921000}, {1.0}},
        {3, {0.7930000}, {1.0}},
    };
}

/// cc-pVDZ-RI for boron (Z=5)
/// Data from BSE: Weigend, Köhn, Hättig, JCP 116, 3175 (2002)
std::vector<AuxShellData> cc_pvdz_ri_boron() {
    return {
        {0, {95.880000}, {1.0}},
        {0, {17.260000}, {1.0}},
        {0, {4.7360000}, {1.0}},
        {0, {1.2730000}, {1.0}},
        {0, {0.3011000}, {1.0}},
        {1, {11.400000}, {1.0}},
        {1, {2.5380000}, {1.0}},
        {1, {0.7515000}, {1.0}},
        {1, {0.2191000}, {1.0}},
        {2, {3.5320000}, {1.0}},
        {2, {0.7904000}, {1.0}},
        {2, {0.2301000}, {1.0}},
        {3, {1.0730000}, {1.0}},
    };
}

/// cc-pVDZ-RI for neon (Z=10)
/// Data from BSE: Weigend, Köhn, Hättig, JCP 116, 3175 (2002)
std::vector<AuxShellData> cc_pvdz_ri_neon() {
    return {
        {0, {540.20000}, {1.0}},
        {0, {103.4000}, {1.0}},
        {0, {27.580000}, {1.0}},
        {0, {7.9810000}, {1.0}},
        {0, {1.8840000}, {1.0}},
        {1, {64.200000}, {1.0}},
        {1, {15.210000}, {1.0}},
        {1, {4.7110000}, {1.0}},
        {1, {1.3210000}, {1.0}},
        {2, {22.240000}, {1.0}},
        {2, {5.2030000}, {1.0}},
        {2, {1.4680000}, {1.0}},
        {3, {6.4100000}, {1.0}},
    };
}

/// cc-pVDZ-RI for sodium (Z=11)
/// Data from BSE: Weigend, Köhn, Hättig, JCP 116, 3175 (2002)
/// TODO: Verify exponents against latest BSE revision
std::vector<AuxShellData> cc_pvdz_ri_sodium() {
    return {
        {0, {46.030000}, {1.0}},
        {0, {10.050000}, {1.0}},
        {0, {2.8930000}, {1.0}},
        {0, {0.8244000}, {1.0}},
        {0, {0.1929000}, {1.0}},
        {1, {10.290000}, {1.0}},
        {1, {2.3570000}, {1.0}},
        {1, {0.6290000}, {1.0}},
        {1, {0.1534000}, {1.0}},
        {2, {2.4940000}, {1.0}},
        {2, {0.5526000}, {1.0}},
        {2, {0.1464000}, {1.0}},
        {3, {0.5260000}, {1.0}},
    };
}

/// cc-pVDZ-RI for magnesium (Z=12)
/// Data from BSE: Weigend, Köhn, Hättig, JCP 116, 3175 (2002)
/// TODO: Verify exponents against latest BSE revision
std::vector<AuxShellData> cc_pvdz_ri_magnesium() {
    return {
        {0, {76.170000}, {1.0}},
        {0, {15.460000}, {1.0}},
        {0, {4.3370000}, {1.0}},
        {0, {1.2140000}, {1.0}},
        {0, {0.2889000}, {1.0}},
        {1, {15.300000}, {1.0}},
        {1, {3.5600000}, {1.0}},
        {1, {0.9702000}, {1.0}},
        {1, {0.2476000}, {1.0}},
        {2, {3.1580000}, {1.0}},
        {2, {0.7259000}, {1.0}},
        {2, {0.2051000}, {1.0}},
        {3, {0.8430000}, {1.0}},
    };
}

/// cc-pVDZ-RI for aluminium (Z=13)
/// Data from BSE: Weigend, Köhn, Hättig, JCP 116, 3175 (2002)
/// TODO: Verify exponents against latest BSE revision
std::vector<AuxShellData> cc_pvdz_ri_aluminium() {
    return {
        {0, {109.8000}, {1.0}},
        {0, {20.450000}, {1.0}},
        {0, {5.6540000}, {1.0}},
        {0, {1.5710000}, {1.0}},
        {0, {0.3718000}, {1.0}},
        {1, {35.050000}, {1.0}},
        {1, {6.9100000}, {1.0}},
        {1, {1.9200000}, {1.0}},
        {1, {0.5290000}, {1.0}},
        {2, {4.6490000}, {1.0}},
        {2, {1.0240000}, {1.0}},
        {2, {0.2923000}, {1.0}},
        {3, {1.2700000}, {1.0}},
    };
}

/// cc-pVDZ-RI for silicon (Z=14)
/// Data from BSE: Weigend, Köhn, Hättig, JCP 116, 3175 (2002)
/// TODO: Verify exponents against latest BSE revision
std::vector<AuxShellData> cc_pvdz_ri_silicon() {
    return {
        {0, {146.3000}, {1.0}},
        {0, {27.200000}, {1.0}},
        {0, {7.5330000}, {1.0}},
        {0, {2.0990000}, {1.0}},
        {0, {0.4948000}, {1.0}},
        {1, {45.870000}, {1.0}},
        {1, {9.2690000}, {1.0}},
        {1, {2.6280000}, {1.0}},
        {1, {0.7404000}, {1.0}},
        {2, {6.1590000}, {1.0}},
        {2, {1.3920000}, {1.0}},
        {2, {0.3999000}, {1.0}},
        {3, {1.6200000}, {1.0}},
    };
}

/// cc-pVDZ-RI for phosphorus (Z=15)
/// Data from BSE: Weigend, Köhn, Hättig, JCP 116, 3175 (2002)
std::vector<AuxShellData> cc_pvdz_ri_phosphorus() {
    return {
        {0, {190.0000}, {1.0}},
        {0, {35.340000}, {1.0}},
        {0, {9.8040000}, {1.0}},
        {0, {2.7390000}, {1.0}},
        {0, {0.6456000}, {1.0}},
        {1, {58.240000}, {1.0}},
        {1, {11.950000}, {1.0}},
        {1, {3.4230000}, {1.0}},
        {1, {0.9755000}, {1.0}},
        {2, {7.9960000}, {1.0}},
        {2, {1.8310000}, {1.0}},
        {2, {0.5291000}, {1.0}},
        {3, {2.0300000}, {1.0}},
    };
}

/// cc-pVDZ-RI for sulfur (Z=16)
/// Data from BSE: Weigend, Köhn, Hättig, JCP 116, 3175 (2002)
std::vector<AuxShellData> cc_pvdz_ri_sulfur() {
    return {
        {0, {238.2000}, {1.0}},
        {0, {44.260000}, {1.0}},
        {0, {12.290000}, {1.0}},
        {0, {3.4380000}, {1.0}},
        {0, {0.8128000}, {1.0}},
        {1, {72.230000}, {1.0}},
        {1, {14.990000}, {1.0}},
        {1, {4.3350000}, {1.0}},
        {1, {1.2430000}, {1.0}},
        {2, {10.120000}, {1.0}},
        {2, {2.3340000}, {1.0}},
        {2, {0.6745000}, {1.0}},
        {3, {2.5050000}, {1.0}},
    };
}

/// cc-pVDZ-RI for chlorine (Z=17)
/// Data from BSE: Weigend, Köhn, Hättig, JCP 116, 3175 (2002)
std::vector<AuxShellData> cc_pvdz_ri_chlorine() {
    return {
        {0, {291.6000}, {1.0}},
        {0, {54.240000}, {1.0}},
        {0, {15.100000}, {1.0}},
        {0, {4.2260000}, {1.0}},
        {0, {1.0020000}, {1.0}},
        {1, {88.170000}, {1.0}},
        {1, {18.440000}, {1.0}},
        {1, {5.3690000}, {1.0}},
        {1, {1.5460000}, {1.0}},
        {2, {12.600000}, {1.0}},
        {2, {2.9230000}, {1.0}},
        {2, {0.8468000}, {1.0}},
        {3, {3.0490000}, {1.0}},
    };
}

// ============================================================================
// cc-pVTZ-RI data (H, C, N, O, F)
// ============================================================================

std::vector<AuxShellData> cc_pvtz_ri_hydrogen() {
    return {
        {0, {19.24000}, {1.0}},
        {0, {4.44800}, {1.0}},
        {0, {1.33600}, {1.0}},
        {0, {0.45360}, {1.0}},
        {1, {5.98400}, {1.0}},
        {1, {1.73300}, {1.0}},
        {1, {0.55400}, {1.0}},
        {2, {3.42400}, {1.0}},
        {2, {0.97600}, {1.0}},
        {3, {2.30200}, {1.0}},
    };
}

std::vector<AuxShellData> cc_pvtz_ri_carbon() {
    return {
        {0, {456.4000}, {1.0}},
        {0, {108.4000}, {1.0}},
        {0, {33.1000}, {1.0}},
        {0, {11.2300}, {1.0}},
        {0, {3.78500}, {1.0}},
        {0, {1.15400}, {1.0}},
        {0, {0.34090}, {1.0}},
        {1, {54.0400}, {1.0}},
        {1, {15.1100}, {1.0}},
        {1, {5.28300}, {1.0}},
        {1, {1.95000}, {1.0}},
        {1, {0.65360}, {1.0}},
        {2, {15.7200}, {1.0}},
        {2, {4.65700}, {1.0}},
        {2, {1.52400}, {1.0}},
        {2, {0.48000}, {1.0}},
        {3, {4.59100}, {1.0}},
        {3, {1.26300}, {1.0}},
        {4, {2.62200}, {1.0}},
    };
}

std::vector<AuxShellData> cc_pvtz_ri_nitrogen() {
    return {
        {0, {625.0000}, {1.0}},
        {0, {151.0000}, {1.0}},
        {0, {46.0400}, {1.0}},
        {0, {15.6900}, {1.0}},
        {0, {5.30500}, {1.0}},
        {0, {1.60600}, {1.0}},
        {0, {0.47300}, {1.0}},
        {1, {73.9600}, {1.0}},
        {1, {20.9300}, {1.0}},
        {1, {7.37400}, {1.0}},
        {1, {2.73800}, {1.0}},
        {1, {0.91250}, {1.0}},
        {2, {21.8800}, {1.0}},
        {2, {6.57200}, {1.0}},
        {2, {2.16700}, {1.0}},
        {2, {0.68000}, {1.0}},
        {3, {6.42100}, {1.0}},
        {3, {1.77200}, {1.0}},
        {4, {3.64500}, {1.0}},
    };
}

std::vector<AuxShellData> cc_pvtz_ri_oxygen() {
    return {
        {0, {812.0000}, {1.0}},
        {0, {199.5000}, {1.0}},
        {0, {60.7400}, {1.0}},
        {0, {20.7800}, {1.0}},
        {0, {7.04900}, {1.0}},
        {0, {2.13500}, {1.0}},
        {0, {0.62800}, {1.0}},
        {1, {96.5500}, {1.0}},
        {1, {27.6500}, {1.0}},
        {1, {9.83200}, {1.0}},
        {1, {3.66700}, {1.0}},
        {1, {1.21800}, {1.0}},
        {2, {28.5500}, {1.0}},
        {2, {8.71600}, {1.0}},
        {2, {2.88200}, {1.0}},
        {2, {0.90200}, {1.0}},
        {3, {8.37400}, {1.0}},
        {3, {2.33100}, {1.0}},
        {4, {4.71800}, {1.0}},
    };
}

std::vector<AuxShellData> cc_pvtz_ri_fluorine() {
    return {
        {0, {1030.000}, {1.0}},
        {0, {255.0000}, {1.0}},
        {0, {78.4200}, {1.0}},
        {0, {26.8200}, {1.0}},
        {0, {9.10900}, {1.0}},
        {0, {2.76100}, {1.0}},
        {0, {0.81200}, {1.0}},
        {1, {123.3000}, {1.0}},
        {1, {35.3400}, {1.0}},
        {1, {12.6600}, {1.0}},
        {1, {4.74100}, {1.0}},
        {1, {1.57500}, {1.0}},
        {2, {36.8100}, {1.0}},
        {2, {11.3700}, {1.0}},
        {2, {3.77900}, {1.0}},
        {2, {1.18000}, {1.0}},
        {3, {10.7300}, {1.0}},
        {3, {3.03600}, {1.0}},
        {4, {6.07000}, {1.0}},
    };
}

// ============================================================================
// def2-SVP-JKFIT data (H, C, N, O, F)
// ============================================================================

std::vector<AuxShellData> def2_svp_jkfit_hydrogen() {
    return {
        {0, {7.34900}, {1.0}},
        {0, {1.63600}, {1.0}},
        {0, {0.38830}, {1.0}},
        {1, {2.29200}, {1.0}},
        {1, {0.60100}, {1.0}},
        {2, {1.77600}, {1.0}},
    };
}

std::vector<AuxShellData> def2_svp_jkfit_carbon() {
    return {
        {0, {163.7000}, {1.0}},
        {0, {29.2200}, {1.0}},
        {0, {7.82200}, {1.0}},
        {0, {2.15000}, {1.0}},
        {0, {0.50470}, {1.0}},
        {1, {19.4300}, {1.0}},
        {1, {4.31700}, {1.0}},
        {1, {1.30400}, {1.0}},
        {1, {0.38420}, {1.0}},
        {2, {6.15500}, {1.0}},
        {2, {1.37300}, {1.0}},
        {2, {0.39770}, {1.0}},
        {3, {1.82900}, {1.0}},
    };
}

std::vector<AuxShellData> def2_svp_jkfit_nitrogen() {
    return {
        {0, {222.6000}, {1.0}},
        {0, {40.8300}, {1.0}},
        {0, {10.7700}, {1.0}},
        {0, {3.01300}, {1.0}},
        {0, {0.70880}, {1.0}},
        {1, {26.3000}, {1.0}},
        {1, {6.0000}, {1.0}},
        {1, {1.81800}, {1.0}},
        {1, {0.52350}, {1.0}},
        {2, {8.6600}, {1.0}},
        {2, {1.93700}, {1.0}},
        {2, {0.55810}, {1.0}},
        {3, {2.58300}, {1.0}},
    };
}

std::vector<AuxShellData> def2_svp_jkfit_oxygen() {
    return {
        {0, {289.4000}, {1.0}},
        {0, {54.1400}, {1.0}},
        {0, {14.3400}, {1.0}},
        {0, {4.04700}, {1.0}},
        {0, {0.95580}, {1.0}},
        {1, {34.3200}, {1.0}},
        {1, {7.93700}, {1.0}},
        {1, {2.42700}, {1.0}},
        {1, {0.68350}, {1.0}},
        {2, {11.6300}, {1.0}},
        {2, {2.64500}, {1.0}},
        {2, {0.75270}, {1.0}},
        {3, {3.37000}, {1.0}},
    };
}

std::vector<AuxShellData> def2_svp_jkfit_fluorine() {
    return {
        {0, {368.6000}, {1.0}},
        {0, {69.7100}, {1.0}},
        {0, {18.4200}, {1.0}},
        {0, {5.25600}, {1.0}},
        {0, {1.24700}, {1.0}},
        {1, {43.8100}, {1.0}},
        {1, {10.2500}, {1.0}},
        {1, {3.14800}, {1.0}},
        {1, {0.88470}, {1.0}},
        {2, {14.9500}, {1.0}},
        {2, {3.46800}, {1.0}},
        {2, {0.98000}, {1.0}},
        {3, {4.30500}, {1.0}},
    };
}

// ============================================================================
// def2-TZVP-JKFIT data (H, C, N, O, F)
// ============================================================================

std::vector<AuxShellData> def2_tzvp_jkfit_hydrogen() {
    return {
        {0, {16.0600}, {1.0}},
        {0, {4.04100}, {1.0}},
        {0, {1.24200}, {1.0}},
        {0, {0.41800}, {1.0}},
        {1, {5.04500}, {1.0}},
        {1, {1.53900}, {1.0}},
        {1, {0.49300}, {1.0}},
        {2, {3.07200}, {1.0}},
        {2, {0.87600}, {1.0}},
        {3, {2.06000}, {1.0}},
    };
}

std::vector<AuxShellData> def2_tzvp_jkfit_carbon() {
    return {
        {0, {399.8000}, {1.0}},
        {0, {95.2000}, {1.0}},
        {0, {29.2100}, {1.0}},
        {0, {9.92000}, {1.0}},
        {0, {3.34600}, {1.0}},
        {0, {1.01500}, {1.0}},
        {0, {0.30050}, {1.0}},
        {1, {46.7800}, {1.0}},
        {1, {12.9600}, {1.0}},
        {1, {4.56200}, {1.0}},
        {1, {1.68300}, {1.0}},
        {1, {0.56450}, {1.0}},
        {2, {13.6300}, {1.0}},
        {2, {4.04300}, {1.0}},
        {2, {1.33000}, {1.0}},
        {2, {0.41900}, {1.0}},
        {3, {3.97200}, {1.0}},
        {3, {1.09300}, {1.0}},
        {4, {2.27000}, {1.0}},
    };
}

std::vector<AuxShellData> def2_tzvp_jkfit_nitrogen() {
    return {
        {0, {545.7000}, {1.0}},
        {0, {132.0000}, {1.0}},
        {0, {40.5700}, {1.0}},
        {0, {13.8400}, {1.0}},
        {0, {4.68900}, {1.0}},
        {0, {1.42200}, {1.0}},
        {0, {0.41900}, {1.0}},
        {1, {63.5600}, {1.0}},
        {1, {17.8200}, {1.0}},
        {1, {6.32600}, {1.0}},
        {1, {2.35600}, {1.0}},
        {1, {0.78800}, {1.0}},
        {2, {18.820}, {1.0}},
        {2, {5.6660}, {1.0}},
        {2, {1.8760}, {1.0}},
        {2, {0.59100}, {1.0}},
        {3, {5.54000}, {1.0}},
        {3, {1.53200}, {1.0}},
        {4, {3.14500}, {1.0}},
    };
}

std::vector<AuxShellData> def2_tzvp_jkfit_oxygen() {
    return {
        {0, {707.3000}, {1.0}},
        {0, {174.0000}, {1.0}},
        {0, {53.5800}, {1.0}},
        {0, {18.4000}, {1.0}},
        {0, {6.25200}, {1.0}},
        {0, {1.89800}, {1.0}},
        {0, {0.55900}, {1.0}},
        {1, {83.1700}, {1.0}},
        {1, {23.5500}, {1.0}},
        {1, {8.43800}, {1.0}},
        {1, {3.16100}, {1.0}},
        {1, {1.05600}, {1.0}},
        {2, {24.5600}, {1.0}},
        {2, {7.51200}, {1.0}},
        {2, {2.50200}, {1.0}},
        {2, {0.78600}, {1.0}},
        {3, {7.22500}, {1.0}},
        {3, {2.02000}, {1.0}},
        {4, {4.08800}, {1.0}},
    };
}

std::vector<AuxShellData> def2_tzvp_jkfit_fluorine() {
    return {
        {0, {892.0000}, {1.0}},
        {0, {221.5000}, {1.0}},
        {0, {68.5700}, {1.0}},
        {0, {23.6000}, {1.0}},
        {0, {8.05300}, {1.0}},
        {0, {2.44800}, {1.0}},
        {0, {0.72100}, {1.0}},
        {1, {105.8000}, {1.0}},
        {1, {30.1400}, {1.0}},
        {1, {10.8700}, {1.0}},
        {1, {4.09200}, {1.0}},
        {1, {1.36600}, {1.0}},
        {2, {31.5900}, {1.0}},
        {2, {9.81400}, {1.0}},
        {2, {3.28300}, {1.0}},
        {2, {1.03000}, {1.0}},
        {3, {9.26500}, {1.0}},
        {3, {2.62300}, {1.0}},
        {4, {5.24800}, {1.0}},
    };
}

// ============================================================================
// Lookup tables
// ============================================================================

using ShellDataFactory = std::vector<AuxShellData>(*)();

struct BasisEntry {
    const char* name;
    FittingType type;
    // Map from atomic_number to factory function
    std::unordered_map<int, ShellDataFactory> element_factories;
};

/// @brief Get all basis entries
const std::vector<BasisEntry>& get_basis_entries() {
    static const std::vector<BasisEntry> entries = {
        {"cc-pVDZ-RI", FittingType::RI,
         {{1, cc_pvdz_ri_hydrogen},
          {3, cc_pvdz_ri_lithium}, {4, cc_pvdz_ri_beryllium},
          {5, cc_pvdz_ri_boron},
          {6, cc_pvdz_ri_carbon}, {7, cc_pvdz_ri_nitrogen},
          {8, cc_pvdz_ri_oxygen}, {9, cc_pvdz_ri_fluorine},
          {10, cc_pvdz_ri_neon},
          {11, cc_pvdz_ri_sodium}, {12, cc_pvdz_ri_magnesium},
          {13, cc_pvdz_ri_aluminium}, {14, cc_pvdz_ri_silicon},
          {15, cc_pvdz_ri_phosphorus}, {16, cc_pvdz_ri_sulfur},
          {17, cc_pvdz_ri_chlorine}}},
        {"cc-pVTZ-RI", FittingType::RI,
         {{1, cc_pvtz_ri_hydrogen}, {6, cc_pvtz_ri_carbon},
          {7, cc_pvtz_ri_nitrogen}, {8, cc_pvtz_ri_oxygen},
          {9, cc_pvtz_ri_fluorine}}},
        {"def2-SVP-JKFIT", FittingType::JKFIT,
         {{1, def2_svp_jkfit_hydrogen}, {6, def2_svp_jkfit_carbon},
          {7, def2_svp_jkfit_nitrogen}, {8, def2_svp_jkfit_oxygen},
          {9, def2_svp_jkfit_fluorine}}},
        {"def2-TZVP-JKFIT", FittingType::JKFIT,
         {{1, def2_tzvp_jkfit_hydrogen}, {6, def2_tzvp_jkfit_carbon},
          {7, def2_tzvp_jkfit_nitrogen}, {8, def2_tzvp_jkfit_oxygen},
          {9, def2_tzvp_jkfit_fluorine}}},
    };
    return entries;
}

/// @brief Case-insensitive string comparison
bool iequals(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) !=
            std::tolower(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

}  // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

AuxiliaryBasisSet create_builtin_auxiliary_basis(
    const std::string& name,
    const std::vector<Atom>& atoms) {

    // Find matching basis entry
    const BasisEntry* found = nullptr;
    for (const auto& entry : get_basis_entries()) {
        if (iequals(name, entry.name)) {
            found = &entry;
            break;
        }
    }

    if (!found) {
        throw InvalidArgumentException(
            "Unknown auxiliary basis set: '" + name +
            "'. Use list_builtin_auxiliary_bases() for available bases.");
    }

    std::vector<Shell> all_shells;
    for (Size i = 0; i < atoms.size(); ++i) {
        int z = atoms[i].atomic_number;
        auto it = found->element_factories.find(z);
        if (it == found->element_factories.end()) {
            throw InvalidArgumentException(
                "Auxiliary basis '" + name + "' not available for element Z=" +
                std::to_string(z));
        }

        auto shell_data = it->second();
        for (const auto& sd : shell_data) {
            all_shells.emplace_back(
                sd.angular_momentum,
                atoms[i].position,
                sd.exponents,
                sd.coefficients);
            all_shells.back().set_atom_index(static_cast<Index>(i));
        }
    }

    return AuxiliaryBasisSet(std::move(all_shells), found->type, name);
}

std::vector<std::string> list_builtin_auxiliary_bases() {
    std::vector<std::string> names;
    for (const auto& entry : get_basis_entries()) {
        names.emplace_back(entry.name);
    }
    return names;
}

bool is_builtin_auxiliary_available(
    const std::string& name,
    const std::vector<int>& atomic_numbers) {

    for (const auto& entry : get_basis_entries()) {
        if (iequals(name, entry.name)) {
            for (int z : atomic_numbers) {
                if (entry.element_factories.find(z) ==
                    entry.element_factories.end()) {
                    return false;
                }
            }
            return true;
        }
    }
    return false;
}

}  // namespace libaccint::data
