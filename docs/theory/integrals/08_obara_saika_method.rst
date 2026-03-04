.. _theory-integrals-os:

Obara-Saika Method
==================

Overview
--------

The Obara-Saika (OS) method evaluates Gaussian integrals via recurrence layers
built from low-order seeds. It is one of the most widely used formulations for
one-electron and two-electron classes in Cartesian representation.

Primitive ERI Seed Structure
----------------------------

For primitive quartets `(a,b,c,d)` with exponent sums
:math:`\zeta = \alpha+\beta`, :math:`\eta = \gamma+\delta`, define

.. math::

   \mathbf{P} = \frac{\alpha\mathbf{A}+\beta\mathbf{B}}{\zeta},
   \qquad
   \mathbf{Q} = \frac{\gamma\mathbf{C}+\delta\mathbf{D}}{\eta}

.. math::

   \rho = \frac{\zeta\eta}{\zeta+\eta},
   \qquad
   T = \rho\,\lVert\mathbf{P}-\mathbf{Q}\rVert^2

The seed class is s-type with Boys dependence:

.. math::

   I_{0000}^{(m)} = K_{AB}K_{CD}\,\frac{2\pi^{5/2}}{\zeta\eta\sqrt{\zeta+\eta}}\,F_m(T)

where :math:`K_{AB},K_{CD}` are Gaussian product prefactors.

VRR Construction
----------------

OS builds angular momentum using VRR. A representative component-wise form:

.. math::
   :label: eq-os-vrr

   I_{a_i+1,b,c,d}^{(m)} =
   (P_i-A_i)I_{a,b,c,d}^{(m)}
   + (W_i-P_i)I_{a,b,c,d}^{(m+1)}
   + \frac{a_i}{2\zeta}\left(I_{a_i-1,b,c,d}^{(m)} - \frac{\rho}{\zeta}I_{a_i-1,b,c,d}^{(m+1)}\right)
   + \frac{c_i}{2(\zeta+\eta)} I_{a,b,c_i-1,d}^{(m+1)}

with analogous equations for other centers/components.

Interpretation:

- first two terms are geometric shifts,
- third/fourth terms correct for existing angular momentum on source centers,
- index :math:`m+1` couples recurrence to higher Boys order.

OS Workflow for ERI Blocks
--------------------------

1. Enumerate primitive quartets surviving screening.
2. Compute pair invariants and Boys values up to :math:`m_{\max}`.
3. Build seed tensor :math:`I^{(m)}_{0000}`.
4. Apply VRR to create required intermediate families.
5. Optionally use HRR to transfer momentum to target center arrangement.
6. Contract primitives and store AO shell block.

Implementation-Grade Pseudocode
--------------------------------

.. code-block:: cpp

   void eri_os_shell_quartet(const ShellQuartet& q, Block& out) {
     zero(out);

     for (PrimAB ab : q.ab_primitives()) {
       for (PrimCD cd : q.cd_primitives()) {
         if (!primitive_bound_pass(ab, cd)) continue;

         Invariants inv = build_os_invariants(ab, cd);   // zeta, eta, rho, P, Q, T
         int m_max = required_mmax(q.angular_tuple());

         SmallVec<double> F(m_max + 1);
         boys_eval(m_max, inv.T, boys_cfg, F.data());

         SeedTensor seed = build_os_seed(inv, F);         // I0000^(m)
         TempTensor temp = vrr_build_all_needed(seed, inv, q.angular_tuple());

         // Optional: HRR transfer stage if target layout requires it
         TempTensor target = maybe_hrr_transfer(temp, inv, q.angular_tuple());

         contract_and_accumulate(target, ab, cd, out);
       }
     }

     if (q.output_is_spherical()) apply_cart_to_sph_transform(q, out);
   }

Data Layout and Scheduling
--------------------------

For performance:

- keep `m` as innermost contiguous dimension in seed/VRR tables,
- precompute recurrence coefficients per primitive quartet,
- use tuple-specialized schedule templates for common low `L` classes,
- avoid generic branching inside hot recurrence loops.

GPU mapping often assigns one thread block per shell quartet tile, with shared
memory caching for invariant scalars and Boys slices.

Complexity and Crossover Behavior
---------------------------------

OS cost grows with total angular momentum because:

- required :math:`m_{\max}` increases,
- number of intermediate recurrence nodes increases,
- temporary tensors become larger.

At high momentum or specific contraction patterns, Rys/MD variants may be
preferred (:doc:`10_rys_quadrature_method`, :doc:`11_mcmurchie_davidson_method`).

Numerical Stability Notes
-------------------------

Main error amplifiers:

- inaccurate Boys seeds,
- cancellation in shift terms,
- non-deterministic contraction order.

Mitigations:

- robust Boys piecewise evaluator,
- high precision accumulation,
- deterministic reduction mode for validation.

Validation Checklist
--------------------

1. Low-`L` exact-value comparisons.
2. Random shell-quartet differential tests vs Rys/MD.
3. Symmetry checks under index permutations.
4. Stress sweeps in high `L`, near-coincident, and large-separation regimes.

Cross References
----------------

- RR framework: :doc:`07_recurrence_relations`
- HRR transfer: :doc:`09_head_gordon_pople_method`
- ERI family comparison: :doc:`15_two_electron_eri_algorithms`
