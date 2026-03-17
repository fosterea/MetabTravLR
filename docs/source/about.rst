How it works
=================

SpaceTravLR models gene regulation as a function of **where a cell is**, **who it
talks to**, and **which transcription factors it expresses** — all at once.
The pipeline has seven sequential steps, each grounded in a concrete mathematical
operation and directly implemented in the codebase.

----

Step 1 — Defining the Spatial Neighbourhood
--------------------------------------------

*How much of a neighbour's signal does a cell actually receive?*

For a cell at spatial coordinate :math:`(u, v)` we define its neighbourhood
:math:`\mathcal{N}(u,v)` as the set of all cells :math:`i` whose centroid lies within radius
:math:`r`:

.. math::
   \mathcal{N}(u,v) = \{ i \mid \| \mathbf{x}_i - (u,v) \|_2 \le r \}

Because cells farther away contribute less signal, each neighbour is weighted by a
**2-D Gaussian kernel** :math:`\Omega` that decays smoothly with physical distance:

.. math::
   \Omega(u, v;\, \mathbf{x}_i) =
       \exp\!\left( -\frac{\| \mathbf{x}_i - (u,v) \|_2^2}{2\sigma^2} \right)

:math:`\sigma` is derived from :math:`r` so that a cell at the boundary of the neighbourhood
receives near-zero weight; cells beyond :math:`r` are assigned :math:`\Omega = 0`.

The effective **neighbourhood weight matrix** :math:`W \in \mathbb{R}^{n \times n}` is
therefore sparse: entry :math:`W_{ij} = \Omega(u_j, v_j;\, \mathbf{x}_i)` for
:math:`i \in \mathcal{N}(j)` and zero otherwise.

.. rubric:: Implementation

:func:`~SpaceTravLR.tools.utils.gaussian_kernel_2d` is a JIT-compiled kernel that computes
:math:`\Omega` for a single origin cell against every other cell in one vectorised pass.
It is called inside :func:`~SpaceTravLR.models.parallel_estimators.compute_radius_weights`
to build the per-radius sparse weight rows used throughout training.


Step 2 — Capturing Ligand–Receptor Signalling
----------------------------------------------

*What paracrine signal does a cell collect from its neighbourhood?*

Receptors are membrane-bound and cannot diffuse, so their contribution is modelled with
the local gene expression :math:`R_j^{(k)}` directly.  For each ligand :math:`\mathcal{L}^{(k)}`
the **received ligand signal** at location :math:`j` is the Gaussian-weighted sum over its
neighbourhood:

.. math::
   \mathcal{L}_j^{(k)} =
       \sum_{i \in \mathcal{N}(j)} \Omega(u,v;\,\mathbf{x}_i)\;
       \tilde{\mathcal{L}}_i^{(k)}

where :math:`\tilde{\mathcal{L}}_i^{(k)}` is the (imputed, normalised) ligand expression
at cell :math:`i`.  This assumes radially symmetric diffusion from the source cell.

The **signalling activity** of the :math:`k`-th ligand–receptor pair at location :math:`j` is
then the Hadamard product of incoming ligand signal and local receptor expression:

.. math::
   \text{LRSignal}_{j}^{(k)} =
       \mathcal{L}_j^{(k)} \times R_j^{(k)}

SpaceTravLR distinguishes two diffusion regimes using separate radius values:

* **secreted ligands** — long-range Gaussian diffusion (default :math:`r = 200` µm)
* **contact-range ligands** — short-range Gaussian diffusion (default :math:`r = 30` µm)

Both are stacked into a single received-ligand matrix before training.

.. rubric:: Implementation

:func:`~SpaceTravLR.models.parallel_estimators.received_ligands` computes
:math:`\mathcal{L}_j^{(k)}` for every cell and every ligand simultaneously, grouping
ligands by their diffusion radius (secreted vs. contact-range signalling).
:func:`~SpaceTravLR.models.parallel_estimators.init_received_ligands` populates
``adata.uns["received_ligands"]`` and ``adata.uns["received_ligands_tfl"]`` once at
setup time so that subsequent training and perturbation loops can reuse the precomputed
values.  The element-wise L-R scores (and the analogous ligand-TF scores for NicheNet-style
cross-terms) are assembled by
:meth:`~SpaceTravLR.models.parallel_estimators.SpatialCellularProgramsEstimator.ligands_receptors_interactions`
and
:meth:`~SpaceTravLR.models.parallel_estimators.SpatialCellularProgramsEstimator.ligand_regulators_interactions`.

----

Step 3 — Spatially Varying Regression
--------------------------------------

*How do regulators shape gene expression differently across space?*

SpaceTravLR learns, for every target gene :math:`\mathcal{H}` and every location
:math:`(u, v)` of cell type :math:`c`, a set of **spatially varying regression coefficients**
:math:`\beta(u,v,c)`.  The full prediction model is:

.. math::

   \begin{aligned}
   y(u,v,c) &=
       \beta_0(u,v,c) \\
       &\quad + \underbrace{\sum_{k=1}^{m}
             \beta_{\mathcal{T}_k}(u,v,c)\,\mathrm{TF}_k}_{\text{transcription factors}} \\
       &\quad + \underbrace{\sum_{j=1}^{l}
             \beta_{\mathcal{L}_j \mathcal{R}_j}(u,v,c)\,
             \mathcal{L}_j^{(uv)} R_j}_{\text{L-R pairs}} \\
       &\quad + \underbrace{\sum_{p=1}^{q}
             \beta_{\mathcal{L}_p \mathcal{T}_p}(u,v,c)\,
             \mathcal{L}_p^{(uv)} \mathrm{TF}_p}_{\text{TF–ligand cross-terms}}
   \end{aligned}
The three regressor groups capture complementary regulatory channels:

* **TF terms** — direct transcription-factor regulation within the cell.
* **L-R terms** — paracrine signalling: received ligand :math:`\times` cognate receptor.
* **TF–ligand (TFL) terms** — interaction between a received ligand and a co-expressed TF,
  as identified by NicheNet's ligand–activity scoring.

Because :math:`\beta(u,v,c)` depends on spatial location, the model can discover that the same
TF is strongly activating in one tissue region and nearly silent in another.

.. rubric:: Loss function

The estimator uses **Group Lasso** regularisation to enforce
sparsity across the three regressor groups:

.. math::
   \mathcal{L} =
       \frac{1}{n} \| y - \hat{y} \|_2^2
       + \lambda \sum_{g} \| \boldsymbol{\beta}_g \|_2

where the sum is over regressor groups :math:`g \in \{\text{TF},\, \text{L-R},\, \text{TFL}\}`.
This drives entire groups of uninformative modulators to zero rather than scattering
weight across many small coefficients.

.. rubric:: Neural architecture

The linear seed is refined by a **CellularNicheNetwork** (CNN) or
**CellularViT** (VisionTransformer) that takes a 2D spatial neighbourhood image of
cell-type densities as input, allowing the network to condition the betas on the
cellular microenvironment.

.. rubric:: Implementation

The coefficients are learned by
:class:`~SpaceTravLR.models.parallel_estimators.SpatialCellularProgramsEstimator` using a
spatially-aware convolutional neural network combined with a LASSO regression head, one model
per target gene.  After training, :meth:`~SpaceTravLR.models.parallel_estimators.SpatialCellularProgramsEstimator.get_betas`
extracts the per-cell coefficient matrix, stored in a
:class:`~SpaceTravLR.beta.BetaFrame` (a :class:`pandas.DataFrame` where rows are cells and
columns are regressors, prefixed ``beta_``).

----

Step 4 — Computing Derivatives
-------------------------------------------------------------------

*What is the effect of perturbing a gene on its downstream targets?*


For the ligand terms, we need to compute the partial derivatives by splitting up the ligand interaction coefficient. Assuming that changes in ligand expression and in the paired receptor expression or TF expression are independent, then by the product rule:


.. rubric:: Implementation

:meth:`~SpaceTravLR.beta.BetaFrame.splash` computes the splashed derivatives for each
gene–modulator pair, returning a weighted beta matrix ready for perturbation.
:meth:`~SpaceTravLR.gene_factory.GeneFactory.splash_betas` wraps this for a single gene
using the stored ligand and expression data inside the
:class:`~SpaceTravLR.gene_factory.GeneFactory` context.


Perturbing gene :math:`\mathcal{T}_i` by :math:`\Delta \mathcal{T}_i` induces a
change in any downstream gene :math:`\mathcal{H}` via the partial derivative of the
regression model.

**Direct TF perturbation** — when :math:`\mathcal{T}_i` appears as a TF regressor:

.. math::
   \frac{\partial \mathcal{H}}{\partial \mathcal{T}_i} =
       \beta_{\mathcal{T}_i}(u,v,c)

**Receptor perturbation** — when the perturbed gene encodes receptor :math:`R_i`
(the ligand field is fixed):

.. math::
   \frac{\partial \mathcal{H}}{\partial R_i} =
       \beta_{\mathcal{L}_i \mathcal{R}_i}(u,v,c) \cdot \mathcal{L}_i^{(uv)}

**Ligand perturbation via L-R** — when the perturbed gene encodes ligand
:math:`\mathcal{L}_i` (receptor expression is held fixed — spatial independence):

.. math::
   \frac{\partial \mathcal{H}}{\partial \mathcal{L}_i} =
       \beta_{\mathcal{L}_i \mathcal{R}_i}(u,v,c) \cdot R_i

**TF–ligand (TFL) cross effects** — perturbing a TF :math:`\mathrm{TF}_p` that interacts
with ligand :math:`\mathcal{L}_p`:

.. math::
   \frac{\partial \mathcal{H}}{\partial \mathrm{TF}_p} =
       \beta_{\mathcal{L}_p \mathcal{T}_p}(u,v,c) \cdot \mathcal{L}_p^{(uv)}

For a ligand perturbation via a TFL term:

.. math::
   \frac{\partial \mathcal{H}}{\partial \mathcal{L}_p} =
       \beta_{\mathcal{L}_p \mathcal{T}_p}(u,v,c) \cdot \mathrm{TF}_p

**Combined response** — the total shift :math:`\Delta \mathcal{H}` at cell :math:`(u,v)` of type
:math:`c` when gene :math:`\mathcal{T}_i` is perturbed by :math:`\Delta \mathcal{T}_i`
(summing over direct TF and L-R channels):

.. math::
   \Delta \mathcal{H} =
       \left(
           \beta_{\mathcal{T}_i}(u,v,c)
           + \beta_{\mathcal{L}_i}(u,v,c) \cdot \mathcal{L}_i^{(uv)}
       \right) \Delta \mathcal{T}_i

In matrix form, for all target genes and all cells simultaneously:

.. math::
   \Delta \mathbf{X} = \widetilde{\mathbf{B}} \cdot \Delta \mathbf{x}_0

where :math:`\widetilde{\mathbf{B}} \in \mathbb{R}^{n_{\text{cells}} \times n_{\text{genes}}}`
is the splashed beta matrix and :math:`\Delta \mathbf{x}_0` is the initial expression delta
vector for the perturbed gene.

.. rubric:: Implementation

The four derivative types are computed in parallel by
:func:`~SpaceTravLR.beta.compute_all_derivatives`:

----

Step 5 — Iterative Network Propagation
----------------------------------------

*How do secondary and higher-order effects ripple through the tissue?*

A single perturbation step captures only the *direct* effect on immediate downstream genes.
SpaceTravLR propagates secondary effects through the gene regulatory network over
:math:`n_{\text{prop}}` rounds (default 4):

.. math::
   \Delta \mathbf{x}^{(t+1)} =
       \mathbf{B}^{(t)} \cdot \Delta \mathbf{x}^{(t)}

where :math:`\mathbf{B}^{(t)}` is the full splash-weighted beta matrix at iteration :math:`t`
and :math:`\Delta \mathbf{x}^{(t)}` is the simulated expression-delta vector after :math:`t`
rounds.  The full accumulated expression is clipped at each step:

.. math::
   \mathbf{x}^{(t+1)} = \mathrm{clip}\!\left(
       \mathbf{x}^{(t)} + \Delta \mathbf{x}^{(t+1)},\;
       0,\;
       \mathbf{x}^{(0)}_{\max}
   \right)

At each step:

1. A **new delta** is computed from the current gene expression and the outstanding
   :math:`\Delta \mathbf{x}^{(t)}`.
2. The **ligand field is re-evaluated** from the updated expression matrix, so secondary
   paracrine effects are captured — the Gaussian-weighted integral is recomputed over the
   shifted expression.
3. The expression matrix is **clipped** to :math:`[0,\, \max_{\text{baseline}}]` to prevent
   runaway growth.

.. rubric:: Implementation

:meth:`~SpaceTravLR.gene_factory.GeneFactory.perturb` orchestrates the propagation loop.
A schematic of one propagation step:

.. code-block:: text

   gene_mtx_t  ──►  splashed_betas ──►  delta_{t+1}
       │                                      │
       └──────── re-compute ligands ◄─────────┘
            (weighted by updated expression)

For a genome-wide screen over all TFs, ligands, and receptors in the trained model, use
:meth:`~SpaceTravLR.gene_factory.GeneFactory.genome_screen`:

.. code-block:: python

   gf.genome_screen(
       save_to='./ko_results',
       mode='knockout',        # or 'overexpression'
       n_propagation=4,
   )

Perturbation results for individual genes can also be run in parallel batches via
:meth:`~SpaceTravLR.gene_factory.GeneFactory.perturb_batch`.

----

Step 6 — Perturbation Modes
-----------------------------

SpaceTravLR models **any** continuous shift in gene expression, not just binary knockouts.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Mode
     - How to invoke
   * - **Knockout** (KO)
     - ``gf.perturb(target, gene_expr=0)`` — sets expression to zero
   * - **Overexpression** (OE)
     - ``gf.perturb(target, gene_expr=X)`` — sets expression to :math:`X > 0`
   * - **Partial KO**
     - ``gene_expr`` between 0 and baseline; relative reduction is modelled
   * - **Combinatorial**
     - ``target=['GeneA','GeneB'], gene_expr=[0, 5.0]``

The set of genes that can be targeted — all TFs, secreted ligands, and receptors in the
trained model — is returned by
:meth:`~SpaceTravLR.gene_factory.GeneFactory.possible_targets`.

