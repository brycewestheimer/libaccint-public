# Basis Set Attribution

The basis set data files in this directory are derived from the
[Basis Set Exchange (BSE)](https://www.basissetexchange.org/) using the
`basis_set_exchange` Python package (v0.9+).

**Download date:** February 11, 2026
**Element coverage:** H–Kr (Z=1–36) where defined for each basis set

## Data Format

Files use the QCSchema JSON format as defined by the
[MolSSI QCSchema](https://molssi-qc-schema.readthedocs.io/) project.

## Basis Sets

### Pople Basis Sets

| File | Basis Set | Elements | Reference |
|------|-----------|----------|-----------|
| `sto-3g.json` | STO-3G | 36 (H–Kr) | Hehre et al., JCP 51, 2657 (1969) |
| `sto-6g.json` | STO-6G | 36 (H–Kr) | Hehre et al., JCP 51, 2657 (1969) |
| `3-21g.json` | 3-21G | 36 (H–Kr) | Binkley et al., JACS 102, 939 (1980) |
| `6-31g.json` | 6-31G | 36 (H–Kr) | Hehre et al., JCP 56, 2257 (1972) |
| `6-31g_st.json` | 6-31G* | 36 (H–Kr) | Hariharan & Pople, TCA 28, 213 (1973) |
| `6-31g_ss.json` | 6-31G** | 36 (H–Kr) | Hariharan & Pople, TCA 28, 213 (1973) |
| `6-31+g_st.json` | 6-31+G* | 18 (H–Ar) | Clark et al., JCC 4, 294 (1983) |
| `6-31++g_ss.json` | 6-31++G** | 18 (H–Ar) | Clark et al., JCC 4, 294 (1983) |
| `6-311g.json` | 6-311G | 26 | McLean & Chandler, JCP 72, 5639 (1980) |
| `6-311g_st.json` | 6-311G* | 26 | McLean & Chandler, JCP 72, 5639 (1980) |
| `6-311g_ss.json` | 6-311G** | 26 | McLean & Chandler, JCP 72, 5639 (1980) |
| `6-311+g_st.json` | 6-311+G* | 20 (H–Ca) | McLean & Chandler, JCP 72, 5639 (1980) |
| `6-311+g_ss.json` | 6-311+G** | 20 (H–Ca) | McLean & Chandler, JCP 72, 5639 (1980) |
| `6-311++g_ss.json` | 6-311++G** | 19 | McLean & Chandler, JCP 72, 5639 (1980) |

### Dunning Correlation-Consistent Basis Sets

| File | Basis Set | Elements | Reference |
|------|-----------|----------|-----------|
| `cc-pvdz.json` | cc-pVDZ | 35 (H–Kr, excl. K) | Dunning, JCP 90, 1007 (1989) |
| `cc-pvtz.json` | cc-pVTZ | 35 (H–Kr, excl. K) | Dunning, JCP 90, 1007 (1989) |
| `cc-pvqz.json` | cc-pVQZ | 35 (H–Kr, excl. K) | Dunning, JCP 90, 1007 (1989) |
| `cc-pv5z.json` | cc-pV5Z | 35 (H–Kr, excl. K) | Dunning, JCP 90, 1007 (1989) |
| `aug-cc-pvdz.json` | aug-cc-pVDZ | 34 (H–Kr, excl. K,Ca) | Kendall et al., JCP 96, 6796 (1992) |
| `aug-cc-pvtz.json` | aug-cc-pVTZ | 34 (H–Kr, excl. K,Ca) | Kendall et al., JCP 96, 6796 (1992) |
| `aug-cc-pvqz.json` | aug-cc-pVQZ | 34 (H–Kr, excl. K,Ca) | Kendall et al., JCP 96, 6796 (1992) |
| `aug-cc-pv5z.json` | aug-cc-pV5Z | 34 (H–Kr, excl. K,Ca) | Kendall et al., JCP 96, 6796 (1992) |

### Karlsruhe def2 Basis Sets

| File | Basis Set | Elements | Reference |
|------|-----------|----------|-----------|
| `def2-svp.json` | def2-SVP | 36 (H–Kr) | Weigend & Ahlrichs, PCCP 7, 3297 (2005) |
| `def2-svpd.json` | def2-SVPD | 36 (H–Kr) | Rappoport & Furche, JCP 133, 134105 (2010) |
| `def2-tzvp.json` | def2-TZVP | 36 (H–Kr) | Weigend & Ahlrichs, PCCP 7, 3297 (2005) |
| `def2-tzvpd.json` | def2-TZVPD | 36 (H–Kr) | Rappoport & Furche, JCP 133, 134105 (2010) |
| `def2-tzvpp.json` | def2-TZVPP | 36 (H–Kr) | Weigend & Ahlrichs, PCCP 7, 3297 (2005) |
| `def2-tzvppd.json` | def2-TZVPPD | 36 (H–Kr) | Rappoport & Furche, JCP 133, 134105 (2010) |
| `def2-qzvp.json` | def2-QZVP | 36 (H–Kr) | Weigend & Ahlrichs, PCCP 7, 3297 (2005) |
| `def2-qzvpd.json` | def2-QZVPD | 36 (H–Kr) | Rappoport & Furche, JCP 133, 134105 (2010) |
| `def2-qzvpp.json` | def2-QZVPP | 36 (H–Kr) | Weigend & Ahlrichs, PCCP 7, 3297 (2005) |
| `def2-qzvppd.json` | def2-QZVPPD | 36 (H–Kr) | Rappoport & Furche, JCP 133, 134105 (2010) |

### Auxiliary / Fitting Basis Sets

| File | Basis Set | Elements | Reference |
|------|-----------|----------|-----------|
| `cc-pvtz-jkfit.json` | cc-pVTZ-JKFIT | 16 (H–Ar, partial) | Weigend, PCCP 4, 4285 (2002) |
| `cc-pvqz-jkfit.json` | cc-pVQZ-JKFIT | 16 (H–Ar, partial) | Weigend, PCCP 4, 4285 (2002) |
| `cc-pv5z-jkfit.json` | cc-pV5Z-JKFIT | 16 (H–Ar, partial) | Weigend, PCCP 4, 4285 (2002) |
| `def2-universal-jkfit.json` | def2-UNIVERSAL-JKFIT | 36 (H–Kr) | Weigend, PCCP 8, 1057 (2006) |
| `cc-pvdz-rifit.json` | cc-pVDZ-RIFIT | 24 | Weigend et al., JCP 116, 3175 (2002) |
| `cc-pvtz-rifit.json` | cc-pVTZ-RIFIT | 34 | Weigend et al., JCP 116, 3175 (2002) |
| `cc-pvqz-rifit.json` | cc-pVQZ-RIFIT | 34 | Weigend et al., JCP 116, 3175 (2002) |
| `cc-pv5z-rifit.json` | cc-pV5Z-RIFIT | 34 | Weigend et al., JCP 116, 3175 (2002) |
| `def2-svp-rifit.json` | def2-SVP-RIFIT | 36 (H–Kr) | Weigend, PCCP 8, 1057 (2006) |
| `def2-tzvp-rifit.json` | def2-TZVP-RIFIT | 36 (H–Kr) | Weigend, PCCP 8, 1057 (2006) |
| `def2-qzvp-rifit.json` | def2-QZVP-RIFIT | 26 | Weigend, PCCP 8, 1057 (2006) |

## Basis Set Exchange Citation

The Basis Set Exchange is maintained by the Molecular Sciences Software
Institute (MolSSI). If you use these basis sets, please cite:

B.P. Pritchard, D. Altarawy, B. Didier, T.D. Gibson, T.L. Windus,
"A New Basis Set Exchange: An Open, Up-to-date Resource for the
Molecular Sciences Community",
*J. Chem. Inf. Model.* **59**, 4814-4820 (2019).
DOI: [10.1021/acs.jcim.9b00725](https://doi.org/10.1021/acs.jcim.9b00725)

## Download Script

Basis sets were downloaded using `scripts/download_basis_sets.py` which uses
the `basis_set_exchange` Python package. To re-download or update:

```bash
pip install basis_set_exchange
python scripts/download_basis_sets.py
```
