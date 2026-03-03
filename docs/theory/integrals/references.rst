.. _theory-integrals-references:

References and Reading Map
==========================

Canonical Primary Papers
------------------------

1. S. Obara and A. Saika, *J. Chem. Phys.* **84** (1986), 3963.
   Efficient recursive computation of molecular integrals over Cartesian
   Gaussian functions.

2. M. Head-Gordon and J. A. Pople, *J. Chem. Phys.* **89** (1988), 5777.
   Two-electron Gaussian integral and derivative evaluation via recurrence
   transfer strategies.

3. J. Rys, M. Dupuis, and H. F. King, *J. Comput. Chem.* **4** (1983), 154.
   Quadrature-based computation of electron repulsion integrals.

4. L. E. McMurchie and E. R. Davidson, *J. Comput. Phys.* **26** (1978), 218.
   One- and two-electron integrals over Cartesian Gaussian functions using
   Hermite expansions.

Foundational Textbooks
----------------------

5. T. Helgaker, P. Jorgensen, and J. Olsen,
   *Molecular Electronic-Structure Theory*, Wiley, 2000.

6. A. Szabo and N. S. Ostlund,
   *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory*, Dover.

Density Fitting and Low-Rank Methods
------------------------------------

7. N. H. F. Beebe and J. Linderberg, *Int. J. Quantum Chem.* **12** (1977), 683.
   Simplification and approximation ideas leading to reduced-rank treatments.

8. O. Vahtras, J. Almlof, and M. W. Feyereisen, *Chem. Phys. Lett.* **213** (1993), 514.
   Integral approximations for LCAO-SCF calculations (early RI framework).

9. H. Koch, A. M. Sanchez de Meras, and T. B. Pedersen,
   *J. Chem. Phys.* **118** (2003), 9481.
   Reduced-scaling electronic-structure formulations using Cholesky decomposition.

Method-to-Chapter Map
---------------------

- Boys functions: :doc:`05_boys_functions`
- OS and HGP: :doc:`08_obara_saika_method`, :doc:`09_head_gordon_pople_method`
- Rys quadrature: :doc:`10_rys_quadrature_method`
- McMurchie-Davidson: :doc:`11_mcmurchie_davidson_method`
- Rotated-axis and hybrid variants: :doc:`12_rotated_axis_method`, :doc:`13_rotated_axis_mcmurchie_davidson`
- TRn/fused transfer: :doc:`14_trn_fused_hrn_method`
- Density fitting and Cholesky: :doc:`19_density_fitting`, :doc:`20_cholesky_decomposition`

Recommended Reading Order
-------------------------

1. Foundations: chapters :doc:`00_overview` to :doc:`07_recurrence_relations`.
2. Core ERI methods: chapters :doc:`08_obara_saika_method` to :doc:`15_two_electron_eri_algorithms`.
3. Production methods: chapters :doc:`16_screening_and_bounds` to :doc:`24_method_selection_guide`.

Citation Infrastructure
-----------------------

For long-term maintenance, migrate these references to a shared BibTeX database
(`sphinxcontrib-bibtex`) and replace manual lists with chapter citation keys.
