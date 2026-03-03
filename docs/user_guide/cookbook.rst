.. _cookbook:

Cookbook
========

This cookbook provides ready-to-use code snippets for common tasks.

One-Electron Integrals
----------------------

Core Hamiltonian Matrix
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>

   using namespace libaccint;

   std::vector<Real> compute_core_hamiltonian(Engine& engine,
                                               const std::vector<data::Atom>& atoms) {
       Size nbf = engine.basis().n_basis_functions();

       std::vector<Real> T, V;
       engine.compute(OneElectronOperator(Operator::kinetic()), T);

       PointChargeParams charges;
       for (const auto& atom : atoms) {
           charges.x.push_back(atom.position.x);
           charges.y.push_back(atom.position.y);
           charges.z.push_back(atom.position.z);
           charges.charge.push_back(static_cast<Real>(atom.atomic_number));
       }
       engine.compute(OneElectronOperator(Operator::nuclear(charges)), V);

       std::vector<Real> H(nbf * nbf);
       for (Size i = 0; i < nbf * nbf; ++i) {
           H[i] = T[i] + V[i];
       }
       return H;
   }

Symmetric Orthogonalization Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <Eigen/Dense>

   Eigen::MatrixXd compute_orthogonalizer(const std::vector<Real>& S_vec, Size nbf) {
       Eigen::Map<const Eigen::MatrixXd> S(S_vec.data(), nbf, nbf);

       Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(S);
       return solver.eigenvectors() *
              solver.eigenvalues().array().rsqrt().matrix().asDiagonal() *
              solver.eigenvectors().transpose();
   }

Two-Electron Integrals
----------------------

Fock Matrix from Density
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   std::pair<std::vector<Real>, std::vector<Real>>
   build_jk(Engine& engine, const std::vector<Real>& D) {
       consumers::FockBuilder fock(engine.basis());
       fock.set_density(D);
       engine.compute(Operator::coulomb(), fock);

       return {fock.get_coulomb(), fock.get_exchange()};
   }

   std::vector<Real> build_fock(Engine& engine,
                                 const std::vector<Real>& H_core,
                                 const std::vector<Real>& D) {
       auto [J, K] = build_jk(engine, D);

       Size n = H_core.size();
       std::vector<Real> F(n);
       for (Size i = 0; i < n; ++i) {
           F[i] = H_core[i] + 2.0 * J[i] - K[i];
       }
       return F;
   }

Nuclear Repulsion Energy
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   Real nuclear_repulsion(const std::vector<data::Atom>& atoms) {
       Real E = 0.0;
       for (size_t i = 0; i < atoms.size(); ++i) {
           for (size_t j = i + 1; j < atoms.size(); ++j) {
               Real dx = atoms[i].position.x - atoms[j].position.x;
               Real dy = atoms[i].position.y - atoms[j].position.y;
               Real dz = atoms[i].position.z - atoms[j].position.z;
               Real r = std::sqrt(dx*dx + dy*dy + dz*dz);
               E += Real(atoms[i].atomic_number * atoms[j].atomic_number) / r;
           }
       }
       return E;
   }

SCF Calculations
----------------

Initial Density Guess (Core Hamiltonian)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   Eigen::MatrixXd initial_density_core(const Eigen::MatrixXd& H,
                                         const Eigen::MatrixXd& X,
                                         int n_occ) {
       Eigen::MatrixXd Hp = X.transpose() * H * X;
       Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Hp);
       Eigen::MatrixXd C = X * solver.eigenvectors();
       return 2.0 * C.leftCols(n_occ) * C.leftCols(n_occ).transpose();
   }

SCF Energy
~~~~~~~~~~

.. code-block:: cpp

   Real scf_energy(const std::vector<Real>& D,
                   const std::vector<Real>& H,
                   const std::vector<Real>& F,
                   Size nbf) {
       Real E = 0.0;
       for (Size i = 0; i < nbf * nbf; ++i) {
           E += 0.5 * D[i] * (H[i] + F[i]);
       }
       return E;
   }

DIIS Extrapolation
~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   class DIIS {
       std::vector<Eigen::MatrixXd> fock_history;
       std::vector<Eigen::MatrixXd> error_history;
       size_t max_vectors = 6;

   public:
       void add(const Eigen::MatrixXd& F, const Eigen::MatrixXd& D,
                const Eigen::MatrixXd& S) {
           // Error: [F, D*S] = FDS - SDF
           Eigen::MatrixXd DS = D * S;
           Eigen::MatrixXd error = F * DS - DS.transpose() * F;

           fock_history.push_back(F);
           error_history.push_back(error);

           if (fock_history.size() > max_vectors) {
               fock_history.erase(fock_history.begin());
               error_history.erase(error_history.begin());
           }
       }

       Eigen::MatrixXd extrapolate() {
           size_t n = fock_history.size();
           if (n < 2) return fock_history.back();

           // Build B matrix
           Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n + 1, n + 1);
           for (size_t i = 0; i < n; ++i) {
               for (size_t j = 0; j < n; ++j) {
                   B(i, j) = (error_history[i].array() *
                              error_history[j].array()).sum();
               }
               B(i, n) = -1.0;
               B(n, i) = -1.0;
           }

           Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + 1);
           rhs(n) = -1.0;

           Eigen::VectorXd c = B.colPivHouseholderQr().solve(rhs);

           Eigen::MatrixXd F = Eigen::MatrixXd::Zero(
               fock_history[0].rows(), fock_history[0].cols());
           for (size_t i = 0; i < n; ++i) {
               F += c(i) * fock_history[i];
           }
           return F;
       }
   };

Basis Set Handling
------------------

Load Multiple Basis Sets
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   BasisSet load_mixed_basis(const std::vector<data::Atom>& atoms,
                              const std::string& default_basis,
                              const std::map<int, std::string>& element_basis) {
       std::vector<Shell> all_shells;

       for (const auto& atom : atoms) {
           std::string basis_name = default_basis;
           if (element_basis.count(atom.atomic_number)) {
               basis_name = element_basis.at(atom.atomic_number);
           }

           // Load basis for this element
           auto element_shells = data::load_element_shells(basis_name, atom);
           all_shells.insert(all_shells.end(),
                            element_shells.begin(), element_shells.end());
       }

       return BasisSet(all_shells);
   }

   // Usage:
   // std::map<int, std::string> mixed = {{6, "cc-pvdz"}, {1, "sto-3g"}};
   // auto basis = load_mixed_basis(atoms, "sto-3g", mixed);

GPU Optimization
----------------

Benchmark CPU vs GPU
~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <chrono>

   struct BenchmarkResult {
       double cpu_ms;
       double gpu_ms;
       double speedup;
   };

   BenchmarkResult benchmark_fock_build(Engine& engine,
                                         const std::vector<Real>& D) {
       consumers::FockBuilder fock(engine.basis());
       fock.set_density(D);

       auto time_build = [&](BackendHint hint) {
           fock.reset();
           auto start = std::chrono::high_resolution_clock::now();
           engine.compute(Operator::coulomb(), fock, hint);
           auto end = std::chrono::high_resolution_clock::now();
           return std::chrono::duration<double, std::milli>(end - start).count();
       };

       BenchmarkResult result;
       result.cpu_ms = time_build(BackendHint::ForceCPU);

       if (engine.gpu_available()) {
           // Warmup
           time_build(BackendHint::ForceGPU);
           result.gpu_ms = time_build(BackendHint::ForceGPU);
           result.speedup = result.cpu_ms / result.gpu_ms;
       } else {
           result.gpu_ms = -1;
           result.speedup = 0;
       }

       return result;
   }

Python Recipes
--------------

NumPy-based SCF
~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from scipy.linalg import eigh
   import libaccint

   def rhf(atoms, basis_name="sto-3g", max_iter=50, tol=1e-8):
       """Minimal RHF implementation."""
       basis = libaccint.basis_set(basis_name, atoms)
       engine = libaccint.Engine(basis)
       nbf = basis.n_basis_functions()
       n_occ = sum(a.atomic_number for a in atoms) // 2

       S = engine.compute_overlap_matrix()
       H = engine.compute_kinetic_matrix() + engine.compute_nuclear_matrix(atoms)

       # X = S^(-1/2)
       e, v = eigh(S)
       X = v @ np.diag(1/np.sqrt(e)) @ v.T

       # Initial guess
       e, c = eigh(X.T @ H @ X)
       C = X @ c
       D = 2 * C[:,:n_occ] @ C[:,:n_occ].T

       for i in range(max_iter):
           J, K = libaccint.build_fock(engine, D)
           F = H + 2*J - K
           E = 0.5 * np.sum(D * (H + F))

           e, c = eigh(X.T @ F @ X)
           C = X @ c
           D_new = 2 * C[:,:n_occ] @ C[:,:n_occ].T

           if np.max(np.abs(D_new - D)) < tol:
               return E
           D = D_new

       raise RuntimeError("SCF did not converge")

Visualize Matrices
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   def plot_matrix(M, title="Matrix", cmap="RdBu"):
       """Plot a matrix with colorbar."""
       plt.figure(figsize=(8, 6))
       im = plt.imshow(M, cmap=cmap, aspect='auto')
       plt.colorbar(im)
       plt.title(title)
       plt.xlabel("Column")
       plt.ylabel("Row")
       plt.tight_layout()
       plt.show()

   # Usage
   S = engine.compute_overlap_matrix()
   plot_matrix(S, "Overlap Matrix")

Export to Other Formats
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import json

   def export_integrals(engine, atoms, filename):
       """Export one-electron integrals to JSON."""
       S = engine.compute_overlap_matrix()
       T = engine.compute_kinetic_matrix()
       V = engine.compute_nuclear_matrix(atoms)

       data = {
           "overlap": S.tolist(),
           "kinetic": T.tolist(),
           "nuclear": V.tolist(),
           "n_basis": S.shape[0],
       }

       with open(filename, 'w') as f:
           json.dump(data, f, indent=2)

   # Usage
   export_integrals(engine, atoms, "integrals.json")

Non-Consumer Compute (Phase 14)
-------------------------------

Computing the ERI Tensor
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``compute_eri_tensor`` method computes the full 4-index electron repulsion
integral tensor without requiring a consumer callback.

**C++:**

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>

   using namespace libaccint;

   Engine engine(basis);
   Size nbf = basis.n_basis_functions();

   // Full ERI tensor as flat vector (nbf^4 elements, row-major)
   auto eri = engine.compute_eri_tensor();

   // Access element (i,j|k,l)
   Real value = eri[i * nbf*nbf*nbf + j * nbf*nbf + k * nbf + l];

   // Verify 8-fold permutational symmetry
   assert(std::abs(eri[i*nbf*nbf*nbf + j*nbf*nbf + k*nbf + l] -
                   eri[j*nbf*nbf*nbf + i*nbf*nbf + k*nbf + l]) < 1e-12);

**Python:**

.. code-block:: python

   import numpy as np
   import libaccint

   engine = libaccint.Engine(basis)
   eri = engine.compute_eri_tensor()          # shape: (nbf, nbf, nbf, nbf)

   # Build J and K from the tensor directly
   J = np.einsum('ijkl,kl->ij', eri, D)
   K = np.einsum('ikjl,kl->ij', eri, D)

Parallel Batch Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``compute_batch_parallel`` to compute integrals for multiple quartets
across threads. Each quartet is processed independently and results are
returned as a vector of ``IntegralBuffer`` objects.

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>

   using namespace libaccint;

   Engine engine(basis);

   // Compute all quartets in parallel (0 = auto-detect thread count)
   auto results = engine.compute_all_2e_parallel(
       Operator::coulomb(), /* n_threads = */ 0);

   // Or compute a specific subset
   const auto& quartets = basis.shell_set_quartets();
   auto partial = engine.compute_batch_parallel(
       Operator::coulomb(), quartets, /* n_threads = */ 4);

   // Each result contains integral values and metadata
   for (const auto& buf : results) {
       if (!buf.empty()) {
           // Process integrals...
       }
   }

Schwarz Screening
~~~~~~~~~~~~~~~~~

``compute_batch_screened`` automatically applies Schwarz inequality screening
to skip negligible shell quartets. Screened-out quartets produce empty
``IntegralBuffer`` entries.

.. code-block:: cpp

   #include <libaccint/libaccint.hpp>

   using namespace libaccint;

   Engine engine(basis);

   // Screen with threshold 1e-10
   ScreeningOptions screening;
   screening.threshold = 1e-10;

   auto results = engine.compute_batch_screened(
       Operator::coulomb(), screening);

   // Analyze screening efficiency
   auto stats = Engine::compute_screening_statistics(results);
   std::cout << "Computed: " << stats.n_computed << "\n"
             << "Screened: " << stats.n_screened << "\n"
             << "Ratio:    " << stats.screening_ratio << "\n";
