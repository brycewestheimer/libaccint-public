// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#include <libaccint/libaccint.hpp>
#include <libaccint/consumers/fock_builder.hpp>
#include <libaccint/data/basis_parser.hpp>
#include <libaccint/data/builtin_basis.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace libaccint;

namespace {

namespace linalg {

inline void mat_mul(const std::vector<Real>& A,
                    const std::vector<Real>& B,
                    std::vector<Real>& C,
                    std::size_t n) {
    C.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            const Real a_ik = A[i * n + k];
            for (std::size_t j = 0; j < n; ++j) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

inline std::vector<Real> transpose(const std::vector<Real>& A, std::size_t n) {
    std::vector<Real> At(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            At[j * n + i] = A[i * n + j];
        }
    }
    return At;
}

inline void jacobi_eigen(const std::vector<Real>& mat,
                         std::vector<Real>& eigenvalues,
                         std::vector<Real>& eigenvectors,
                         std::size_t n,
                         Real tol = 1.0e-14,
                         int max_sweeps = 200) {
    std::vector<Real> A = mat;
    eigenvectors.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        eigenvectors[i * n + i] = 1.0;
    }

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        Real off_norm = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                off_norm += A[i * n + j] * A[i * n + j];
            }
        }
        off_norm = std::sqrt(2.0 * off_norm);
        if (off_norm < tol) {
            break;
        }

        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const Real a_pq = A[p * n + q];
                if (std::abs(a_pq) < tol * 1.0e-2) {
                    continue;
                }

                const Real a_pp = A[p * n + p];
                const Real a_qq = A[q * n + q];
                Real theta = 0.0;
                if (std::abs(a_pp - a_qq) < 1.0e-30) {
                    theta = static_cast<Real>(std::numbers::pi_v<double> / 4.0);
                } else {
                    theta = static_cast<Real>(0.5) * std::atan2(static_cast<Real>(2.0) * a_pq,
                                                                a_pp - a_qq);
                }

                const Real c = std::cos(theta);
                const Real s = std::sin(theta);

                for (std::size_t i = 0; i < n; ++i) {
                    const Real a_ip = A[i * n + p];
                    const Real a_iq = A[i * n + q];
                    A[i * n + p] = c * a_ip + s * a_iq;
                    A[i * n + q] = -s * a_ip + c * a_iq;
                }
                for (std::size_t j = 0; j < n; ++j) {
                    const Real a_pj = A[p * n + j];
                    const Real a_qj = A[q * n + j];
                    A[p * n + j] = c * a_pj + s * a_qj;
                    A[q * n + j] = -s * a_pj + c * a_qj;
                }

                for (std::size_t i = 0; i < n; ++i) {
                    const Real v_ip = eigenvectors[i * n + p];
                    const Real v_iq = eigenvectors[i * n + q];
                    eigenvectors[i * n + p] = c * v_ip + s * v_iq;
                    eigenvectors[i * n + q] = -s * v_ip + c * v_iq;
                }
            }
        }
    }

    eigenvalues.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        eigenvalues[i] = A[i * n + i];
    }

    for (std::size_t i = 0; i < n; ++i) {
        std::size_t min_idx = i;
        for (std::size_t j = i + 1; j < n; ++j) {
            if (eigenvalues[j] < eigenvalues[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            std::swap(eigenvalues[i], eigenvalues[min_idx]);
            for (std::size_t k = 0; k < n; ++k) {
                std::swap(eigenvectors[k * n + i], eigenvectors[k * n + min_idx]);
            }
        }
    }
}

}  // namespace linalg

struct CliOptions {
    std::string molecule_spec = "h2o";
    std::string basis_name = "sto-3g";
    std::string units = "bohr";
    std::string backend = "auto";
    std::string two_e_mode = "buffer";
    int charge = 0;
    int max_iter = 100;
    int gpu_slots = 4;
    Real conv_thresh = 1.0e-10;
};

struct MoleculeInput {
    std::string label;
    std::vector<data::Atom> atoms;
};

enum class TwoElectronMode {
    Buffer,
    Consumer,
    Compare,
};

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string trim(std::string s) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    auto it_l = std::find_if(s.begin(), s.end(), not_space);
    if (it_l == s.end()) {
        return "";
    }
    auto it_r = std::find_if(s.rbegin(), s.rend(), not_space).base();
    return std::string(it_l, it_r);
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        out.push_back(item);
    }
    return out;
}

int atomic_number_from_symbol(std::string symbol) {
    symbol = trim(symbol);
    if (symbol.empty()) {
        throw std::invalid_argument("empty atomic symbol");
    }

    const bool numeric = std::all_of(symbol.begin(), symbol.end(), [](unsigned char c) {
        return std::isdigit(c);
    });
    if (numeric) {
        const int z = std::stoi(symbol);
        if (z <= 0) {
            throw std::invalid_argument("atomic number must be positive");
        }
        return z;
    }

    symbol[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(symbol[0])));
    for (std::size_t i = 1; i < symbol.size(); ++i) {
        symbol[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(symbol[i])));
    }

    if (symbol == "H") return 1;
    if (symbol == "He") return 2;
    if (symbol == "Li") return 3;
    if (symbol == "Be") return 4;
    if (symbol == "B") return 5;
    if (symbol == "C") return 6;
    if (symbol == "N") return 7;
    if (symbol == "O") return 8;
    if (symbol == "F") return 9;
    if (symbol == "Ne") return 10;

    throw std::invalid_argument("unsupported atomic symbol in mini-app: " + symbol);
}

Real unit_scale_bohr(const std::string& units_raw) {
    const std::string units = to_lower(units_raw);
    if (units == "bohr") {
        return static_cast<Real>(1.0);
    }
    if (units == "angstrom" || units == "ang") {
        return static_cast<Real>(1.0 / 0.529177210903);
    }
    throw std::invalid_argument("units must be 'bohr' or 'angstrom'");
}

MoleculeInput molecule_from_preset(const std::string& key_lower) {
    if (key_lower == "h2") {
        return {
            "H2",
            {
                {1, {0.0, 0.0, 0.0}},
                {1, {0.0, 0.0, 1.4}},
            },
        };
    }

    if (key_lower == "h2o") {
        return {
            "H2O",
            {
                {8, {0.000000,  0.000000,  0.117176}},
                {1, {0.000000,  1.430665, -0.468706}},
                {1, {0.000000, -1.430665, -0.468706}},
            },
        };
    }

    throw std::invalid_argument("unknown preset molecule: " + key_lower);
}

MoleculeInput parse_molecule_spec(const std::string& spec, const std::string& units) {
    const std::string spec_trim = trim(spec);
    if (spec_trim.empty()) {
        throw std::invalid_argument("molecule specification is empty");
    }

    const std::string spec_lower = to_lower(spec_trim);
    if (spec_lower == "h2" || spec_lower == "h2o") {
        return molecule_from_preset(spec_lower);
    }

    const Real scale = unit_scale_bohr(units);
    std::vector<data::Atom> atoms;
    const auto entries = split(spec_trim, ';');
    for (const auto& raw_entry : entries) {
        std::string entry = trim(raw_entry);
        if (entry.empty()) {
            continue;
        }
        std::replace(entry.begin(), entry.end(), ',', ' ');

        std::istringstream iss(entry);
        std::string symbol;
        Real x = 0.0;
        Real y = 0.0;
        Real z = 0.0;
        if (!(iss >> symbol >> x >> y >> z)) {
            throw std::invalid_argument(
                "could not parse atom entry: '" + raw_entry + "'. Expected 'Element x y z'.");
        }

        const int atomic_number = atomic_number_from_symbol(symbol);
        atoms.push_back({atomic_number, {scale * x, scale * y, scale * z}});
    }

    if (atoms.empty()) {
        throw std::invalid_argument("custom molecule specification produced zero atoms");
    }

    return {"custom", atoms};
}

int electron_count(const std::vector<data::Atom>& atoms, int charge) {
    int total_z = 0;
    for (const auto& atom : atoms) {
        total_z += atom.atomic_number;
    }
    return total_z - charge;
}

PointChargeParams make_nuclear_charges(const std::vector<data::Atom>& atoms) {
    PointChargeParams charges;
    charges.x.reserve(atoms.size());
    charges.y.reserve(atoms.size());
    charges.z.reserve(atoms.size());
    charges.charge.reserve(atoms.size());
    for (const auto& atom : atoms) {
        charges.x.push_back(atom.position.x);
        charges.y.push_back(atom.position.y);
        charges.z.push_back(atom.position.z);
        charges.charge.push_back(static_cast<Real>(atom.atomic_number));
    }
    return charges;
}

Real compute_nuclear_repulsion(const std::vector<data::Atom>& atoms) {
    Real e_nuc = 0.0;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        for (std::size_t j = i + 1; j < atoms.size(); ++j) {
            const Real dx = atoms[i].position.x - atoms[j].position.x;
            const Real dy = atoms[i].position.y - atoms[j].position.y;
            const Real dz = atoms[i].position.z - atoms[j].position.z;
            const Real r = std::sqrt(dx * dx + dy * dy + dz * dz);
            e_nuc += static_cast<Real>(atoms[i].atomic_number * atoms[j].atomic_number) / r;
        }
    }
    return e_nuc;
}

std::vector<Real> symmetric_orthogonalization(const std::vector<Real>& S, std::size_t n) {
    std::vector<Real> s_vals;
    std::vector<Real> U;
    linalg::jacobi_eigen(S, s_vals, U, n);

    std::vector<Real> D_Ut(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const Real inv_sqrt = static_cast<Real>(1.0) / std::sqrt(s_vals[i]);
        for (std::size_t j = 0; j < n; ++j) {
            D_Ut[i * n + j] = inv_sqrt * U[j * n + i];
        }
    }

    std::vector<Real> X;
    linalg::mat_mul(U, D_Ut, X, n);
    return X;
}

void symmetrize_matrix(std::vector<Real>& M, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const Real avg = static_cast<Real>(0.5) * (M[i * n + j] + M[j * n + i]);
            M[i * n + j] = avg;
            M[j * n + i] = avg;
        }
    }
}

void scatter_pair_buffer_symmetric(const IntegralBuffer& buf,
                                   std::vector<Real>& target,
                                   std::size_t nbf) {
    for (Size p = 0; p < buf.n_shell_pairs(); ++p) {
        const auto& meta = buf.pair_meta(p);
        auto data = buf.pair_data(p);

        Size idx = 0;
        for (int a = 0; a < meta.na; ++a) {
            const Size i = meta.fi + static_cast<Index>(a);
            for (int b = 0; b < meta.nb; ++b) {
                const Size j = meta.fj + static_cast<Index>(b);
                const Real value = data[idx++];
                target[i * nbf + j] += value;
                if (i != j) {
                    target[j * nbf + i] += value;
                }
            }
        }
    }
}

std::vector<Real> build_1e_matrix_from_pair_buffers(
    Engine& engine,
    const Operator& op,
    const std::vector<ShellSetPair>& pairs,
    std::size_t nbf,
    BackendHint hint) {

    std::vector<Real> matrix(nbf * nbf, 0.0);
    for (const auto& pair_batch : pairs) {
        IntegralBuffer buf = engine.compute(op, pair_batch, hint);
        scatter_pair_buffer_symmetric(buf, matrix, nbf);
    }
    return matrix;
}

void build_jk_from_quartet_buffers(Engine& engine,
                                   const std::vector<ShellSetQuartet>& quartets,
                                   const std::vector<Real>& D,
                                   std::size_t nbf,
                                   BackendHint hint,
                                   std::vector<Real>& J,
                                   std::vector<Real>& K) {
    J.assign(nbf * nbf, 0.0);
    K.assign(nbf * nbf, 0.0);

    const Operator coulomb = Operator::coulomb();
    for (const auto& quartet_batch : quartets) {
        IntegralBuffer buf = engine.compute(coulomb, quartet_batch, hint);
        for (Size q = 0; q < buf.n_shell_quartets(); ++q) {
            const auto& meta = buf.quartet_meta(q);
            auto data = buf.quartet_data(q);

            Size idx = 0;
            for (int a = 0; a < meta.na; ++a) {
                const Size i = meta.fi + static_cast<Index>(a);
                for (int b = 0; b < meta.nb; ++b) {
                    const Size j = meta.fj + static_cast<Index>(b);
                    for (int c = 0; c < meta.nc; ++c) {
                        const Size k = meta.fk + static_cast<Index>(c);
                        for (int d = 0; d < meta.nd; ++d) {
                            const Size l = meta.fl + static_cast<Index>(d);
                            const Real v = data[idx++];
                            J[i * nbf + j] += D[k * nbf + l] * v;
                            K[i * nbf + k] += D[j * nbf + l] * v;
                        }
                    }
                }
            }
        }
    }

    symmetrize_matrix(J, nbf);
    symmetrize_matrix(K, nbf);
}

void build_jk_from_consumer(Engine& engine,
                            const std::vector<ShellSetQuartet>& quartets,
                            const std::vector<Real>& D,
                            std::size_t nbf,
                            BackendHint hint,
                            std::vector<Real>& J,
                            std::vector<Real>& K) {
    consumers::FockBuilder fock(static_cast<Size>(nbf));
    fock.set_density(D.data(), static_cast<Size>(nbf));

    const Operator coulomb = Operator::coulomb();
    for (const auto& quartet_batch : quartets) {
        engine.compute(coulomb, quartet_batch, fock, hint);
    }

    J = fock.get_coulomb_matrix();
    K = fock.get_exchange_matrix();
}

BackendHint parse_backend_hint(const std::string& raw) {
    const std::string v = to_lower(raw);
    if (v == "auto") return BackendHint::Auto;
    if (v == "force-cpu") return BackendHint::ForceCPU;
    if (v == "prefer-cpu") return BackendHint::PreferCPU;
    if (v == "prefer-gpu") return BackendHint::PreferGPU;
    if (v == "force-gpu") return BackendHint::ForceGPU;
    throw std::invalid_argument("invalid backend: " + raw);
}

TwoElectronMode parse_two_e_mode(const std::string& raw) {
    const std::string v = to_lower(raw);
    if (v == "buffer") return TwoElectronMode::Buffer;
    if (v == "consumer") return TwoElectronMode::Consumer;
    if (v == "compare") return TwoElectronMode::Compare;
    throw std::invalid_argument("invalid --two-e-mode: " + raw);
}

void print_usage(const char* exe) {
    std::cout
        << "Usage: " << exe << " [options]\n\n"
        << "Options:\n"
        << "  --molecule <spec>     Preset ('h2', 'h2o') or inline atoms:\n"
        << "                        \"O 0 0 0.117176; H 0 1.430665 -0.468706; H 0 -1.430665 -0.468706\"\n"
        << "  --units <bohr|angstrom>  Units for custom inline molecule (default: bohr)\n"
        << "  --basis <name>        Basis set name or JSON file path (default: sto-3g)\n"
        << "  --charge <int>        Total molecular charge (default: 0)\n"
        << "  --backend <mode>      auto|force-cpu|prefer-cpu|prefer-gpu|force-gpu\n"
        << "  --two-e-mode <mode>   buffer|consumer|compare\n"
        << "  --max-iter <int>      SCF max iterations (default: 100)\n"
        << "  --conv-thresh <real>  SCF energy threshold (default: 1e-10)\n"
        << "  --gpu-slots <int>     Concurrent GPU execution slots (default: 4)\n"
        << "  --help                Show this help\n\n"
        << "Examples:\n"
        << "  " << exe << " --molecule h2o --basis sto-3g --two-e-mode buffer\n"
        << "  " << exe << " --molecule \"H 0 0 0; H 0 0 1.4\" --basis sto-3g --backend prefer-gpu\n";
}

CliOptions parse_cli(int argc, char** argv) {
    CliOptions opts;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--molecule" || arg == "-m") {
            opts.molecule_spec = require_value(arg);
        } else if (arg == "--basis" || arg == "-b") {
            opts.basis_name = require_value(arg);
        } else if (arg == "--units") {
            opts.units = require_value(arg);
        } else if (arg == "--charge") {
            opts.charge = std::stoi(require_value(arg));
        } else if (arg == "--backend") {
            opts.backend = require_value(arg);
        } else if (arg == "--two-e-mode") {
            opts.two_e_mode = require_value(arg);
        } else if (arg == "--max-iter") {
            opts.max_iter = std::stoi(require_value(arg));
        } else if (arg == "--gpu-slots") {
            opts.gpu_slots = std::stoi(require_value(arg));
        } else if (arg == "--conv-thresh") {
            opts.conv_thresh = static_cast<Real>(std::stod(require_value(arg)));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    return opts;
}

BasisSet build_basis(const std::string& basis_name, const std::vector<data::Atom>& atoms) {
    try {
        return data::create_builtin_basis(basis_name, atoms);
    } catch (const std::exception&) {
        const bool is_json = basis_name.size() >= 5 &&
            basis_name.substr(basis_name.size() - 5) == ".json";
        if (is_json) {
            return data::load_basis_set_from_file(basis_name, atoms);
        }
        return data::load_basis_set(basis_name, atoms);
    }
}

Real max_abs_diff(const std::vector<Real>& a, const std::vector<Real>& b) {
    Real max_v = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        max_v = std::max(max_v, std::abs(a[i] - b[i]));
    }
    return max_v;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions opts = parse_cli(argc, argv);
        const BackendHint hint = parse_backend_hint(opts.backend);
        const TwoElectronMode two_e_mode = parse_two_e_mode(opts.two_e_mode);

        MoleculeInput mol = parse_molecule_spec(opts.molecule_spec, opts.units);
        const int n_electrons = electron_count(mol.atoms, opts.charge);
        if (n_electrons <= 0) {
            throw std::invalid_argument("electron count must be positive after applying charge");
        }
        if ((n_electrons % 2) != 0) {
            throw std::invalid_argument(
                "mini-app supports only closed-shell RHF (even number of electrons)");
        }

        const int n_occ = n_electrons / 2;
        BasisSet basis = build_basis(opts.basis_name, mol.atoms);
        const std::size_t nbf = basis.n_basis_functions();
        if (n_occ > static_cast<int>(nbf)) {
            throw std::invalid_argument("not enough basis functions for occupied orbitals");
        }

        DispatchConfig dispatch_config;
        dispatch_config.n_gpu_slots = static_cast<Size>(opts.gpu_slots);
        Engine engine(basis, dispatch_config);
        const auto& pairs = basis.shell_set_pairs();
        const auto& quartets = basis.shell_set_quartets();

        std::cout << "============================================================\n";
        std::cout << "  LibAccInt Mini-App: Restricted Hartree-Fock (RHF)\n";
        std::cout << "============================================================\n\n";
        std::cout << "Molecule:              " << mol.label << "\n";
        std::cout << "Atoms:                 " << mol.atoms.size() << "\n";
        std::cout << "Charge:                " << opts.charge << "\n";
        std::cout << "Electrons:             " << n_electrons << "\n";
        std::cout << "Basis:                 " << opts.basis_name << "\n";
        std::cout << "Basis functions:       " << nbf << "\n";
        std::cout << "ShellSetPairs:         " << pairs.size() << "\n";
        std::cout << "ShellSetQuartets:      " << quartets.size() << "\n";
        std::cout << "Requested backend:     " << opts.backend << "\n";
        std::cout << "GPU available:         " << (engine.gpu_available() ? "yes" : "no") << "\n";
        std::cout << "GPU slots:             " << opts.gpu_slots << "\n";
        std::cout << "Two-electron mode:     " << opts.two_e_mode << "\n\n";

        PointChargeParams charges = make_nuclear_charges(mol.atoms);

        const Operator op_s = Operator::overlap();
        const Operator op_t = Operator::kinetic();
        const Operator op_v = Operator::nuclear(charges);

        std::vector<Real> S = build_1e_matrix_from_pair_buffers(engine, op_s, pairs, nbf, hint);
        std::vector<Real> T = build_1e_matrix_from_pair_buffers(engine, op_t, pairs, nbf, hint);
        std::vector<Real> V = build_1e_matrix_from_pair_buffers(engine, op_v, pairs, nbf, hint);

        std::vector<Real> H(nbf * nbf, 0.0);
        for (std::size_t i = 0; i < nbf * nbf; ++i) {
            H[i] = T[i] + V[i];
        }

        const std::vector<Real> X = symmetric_orthogonalization(S, nbf);
        const std::vector<Real> Xt = linalg::transpose(X, nbf);

        std::vector<Real> XtH;
        std::vector<Real> H_prime;
        linalg::mat_mul(Xt, H, XtH, nbf);
        linalg::mat_mul(XtH, X, H_prime, nbf);

        std::vector<Real> eps;
        std::vector<Real> C_prime;
        linalg::jacobi_eigen(H_prime, eps, C_prime, nbf);

        std::vector<Real> C;
        linalg::mat_mul(X, C_prime, C, nbf);

        const Real E_nuc = compute_nuclear_repulsion(mol.atoms);
        Real E_old = 0.0;

        std::cout << "Starting SCF iteration (max " << opts.max_iter << " cycles)...\n";
        std::cout << std::string(70, '-') << '\n';
        std::cout << std::setw(5) << "Iter"
                  << std::setw(22) << "E_total (Hartree)"
                  << std::setw(18) << "Delta_E"
                  << std::setw(12) << "Status" << '\n';
        std::cout << std::string(70, '-') << '\n';

        for (int iter = 1; iter <= opts.max_iter; ++iter) {
            std::vector<Real> D(nbf * nbf, 0.0);
            for (std::size_t p = 0; p < nbf; ++p) {
                for (std::size_t q = 0; q < nbf; ++q) {
                    for (int i = 0; i < n_occ; ++i) {
                        D[p * nbf + q] += static_cast<Real>(2.0) *
                                           C[p * nbf + static_cast<std::size_t>(i)] *
                                           C[q * nbf + static_cast<std::size_t>(i)];
                    }
                }
            }

            std::vector<Real> J_buffer;
            std::vector<Real> K_buffer;
            std::vector<Real> J_consumer;
            std::vector<Real> K_consumer;

            if (two_e_mode == TwoElectronMode::Buffer || two_e_mode == TwoElectronMode::Compare) {
                build_jk_from_quartet_buffers(engine, quartets, D, nbf, hint, J_buffer, K_buffer);
            }
            if (two_e_mode == TwoElectronMode::Consumer || two_e_mode == TwoElectronMode::Compare) {
                build_jk_from_consumer(engine, quartets, D, nbf, hint, J_consumer, K_consumer);
            }

            const bool use_consumer_jk =
                (two_e_mode == TwoElectronMode::Consumer || two_e_mode == TwoElectronMode::Compare);
            const std::vector<Real>& J = use_consumer_jk ? J_consumer : J_buffer;
            const std::vector<Real>& K = use_consumer_jk ? K_consumer : K_buffer;

            if (two_e_mode == TwoElectronMode::Compare) {
                const Real dJ = max_abs_diff(J_buffer, J_consumer);
                const Real dK = max_abs_diff(K_buffer, K_consumer);
                std::cout << "  [compare] max|J_buffer-J_consumer|=" << std::scientific
                          << dJ << "  max|K_buffer-K_consumer|=" << dK << std::fixed << "\n";
            }

            std::vector<Real> F(nbf * nbf, 0.0);
            for (std::size_t i = 0; i < nbf * nbf; ++i) {
                F[i] = H[i] + J[i] - static_cast<Real>(0.5) * K[i];
            }

            std::vector<Real> XtF;
            std::vector<Real> F_prime;
            linalg::mat_mul(Xt, F, XtF, nbf);
            linalg::mat_mul(XtF, X, F_prime, nbf);

            linalg::jacobi_eigen(F_prime, eps, C_prime, nbf);
            linalg::mat_mul(X, C_prime, C, nbf);

            Real E_elec = 0.0;
            for (std::size_t p = 0; p < nbf; ++p) {
                for (std::size_t q = 0; q < nbf; ++q) {
                    E_elec += D[p * nbf + q] * (H[p * nbf + q] + F[p * nbf + q]);
                }
            }
            E_elec *= static_cast<Real>(0.5);

            const Real E_total = E_elec + E_nuc;
            const Real delta_E = std::abs(E_total - E_old);

            std::cout << std::setw(5) << iter
                      << std::setw(22) << std::fixed << std::setprecision(10) << E_total
                      << std::setw(18) << std::scientific << std::setprecision(4) << delta_E
                      << std::fixed;

            if (delta_E < opts.conv_thresh && iter > 1) {
                std::cout << std::setw(12) << "CONV" << '\n';
                std::cout << std::string(70, '-') << "\n\n";
                std::cout << "SCF converged in " << iter << " iterations.\n\n";
                std::cout << std::fixed << std::setprecision(10);
                std::cout << "Electronic energy:  " << E_elec << " Hartree\n";
                std::cout << "Nuclear repulsion:  " << E_nuc << " Hartree\n";
                std::cout << "Total RHF energy:   " << E_total << " Hartree\n\n";

                std::cout << "Orbital energies (Hartree):\n";
                for (std::size_t i = 0; i < nbf; ++i) {
                    const bool occ = static_cast<int>(i) < n_occ;
                    std::cout << "  " << std::setw(3) << (i + 1)
                              << ": " << std::setw(14) << std::setprecision(6) << eps[i]
                              << (occ ? "  (occupied)" : "  (virtual)") << '\n';
                }
                std::cout << '\n';
                return 0;
            }

            std::cout << std::setw(12) << "" << '\n';
            E_old = E_total;
        }

        std::cerr << "\nWARNING: SCF did not converge within "
                  << opts.max_iter << " iterations.\n";
        std::cerr << "Last energy: " << std::setprecision(10) << E_old << " Hartree\n";
        return 1;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 2;
    }
}
