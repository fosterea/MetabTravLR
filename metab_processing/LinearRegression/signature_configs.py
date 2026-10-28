"""Single source of truth for the signature-analysis datasets: gene-set lists, per-cell
settings, data paths, and the SLURM partition hint. Shared by the compute script
(`signature_run.py` / `signature_batch.py`), the dispatch notebook, and the viewer
notebooks so compute and plotting never drift."""
from dataclasses import dataclass, field

from metab_processing.metab_travlr_config import DATA_DIR

DBVER = '2024.1.Hs'
CATEGORIES = ('h.all', 'c2.cp.kegg_legacy', 'c2.cp.reactome', 'c2.cp.wikipathways', 'c5.go.bp')
CACHE_ROOT = f'{DATA_DIR}/signature_cache'

# --- T-cell program signatures we PREDICT (pseudo-gene targets) ---
TCELL_TARGETS = [
    'HALLMARK_IL2_STAT5_SIGNALING',
    'KEGG_T_CELL_RECEPTOR_SIGNALING_PATHWAY',
    'GOBP_T_CELL_MEDIATED_CYTOTOXICITY',
    'KEGG_NATURAL_KILLER_CELL_MEDIATED_CYTOTOXICITY',
    'HALLMARK_TNFA_SIGNALING_VIA_NFKB',
    'HALLMARK_INTERFERON_GAMMA_RESPONSE',
    'HALLMARK_INTERFERON_ALPHA_RESPONSE',
    'GOBP_CELL_KILLING',
    'GOBP_GRANZYME_MEDIATED_PROGRAMMED_CELL_DEATH_SIGNALING_PATHWAY',
    'GOBP_LYMPHOCYTE_MEDIATED_IMMUNITY',
]
NEG_CONTROL = ['HALLMARK_PANCREAS_BETA_CELLS', 'HALLMARK_SPERMATOGENESIS']

# Metabolic program signatures also PREDICTED as targets (the melanoma metabolic block),
# added to the cancer datasets alongside the T-cell targets. (The colon list already has
# these.) NOTE: hypoxia/glycolysis/oxphos are also cancer FACTORS -- a gene set can be both a
# predicted target (the cell's own program) and a neighborhood factor (a neighbor's program).
METABOLIC_TARGETS = [
    'HALLMARK_HYPOXIA', 'HALLMARK_GLYCOLYSIS', 'HALLMARK_OXIDATIVE_PHOSPHORYLATION',
    'HALLMARK_FATTY_ACID_METABOLISM', 'HALLMARK_MTORC1_SIGNALING',
]

# Manual (non-MSigDB) negative-control gene sets: {name: [literal gene list]}. Resolved from
# the list directly (not the MSigDB library). Cancers only. The islet/neuroendocrine set below
# complements HALLMARK_PANCREAS_BETA_CELLS but with less overlap with T-cell-expressed genes.
MANUAL_NEG_CONTROLS = {
    'MANUAL_ISLET_NEG_CONTROL': [
        'ABCC8', 'CHGA', 'DCX', 'G6PC2', 'PCSK1', 'PCSK2', 'PKLR', 'SCGN', 'SLC2A2', 'SYT13',
    ],
}

# --- Cancer-associated signatures ADDED AS FACTORS (spatial-neighborhood predictors) ---
CANCER_FACTORS = [
    'HALLMARK_G2M_CHECKPOINT', 'HALLMARK_E2F_TARGETS', 'HALLMARK_ANGIOGENESIS',
    'HALLMARK_P53_PATHWAY', 'HALLMARK_KRAS_SIGNALING_UP', 'HALLMARK_KRAS_SIGNALING_DN',
    'HALLMARK_MYC_TARGETS_V1',
    'HALLMARK_GLYCOLYSIS', 'HALLMARK_HYPOXIA', 'HALLMARK_OXIDATIVE_PHOSPHORYLATION',
    'HALLMARK_CHOLESTEROL_HOMEOSTASIS', 'HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY',
    'KEGG_GLYCOLYSIS_GLUCONEOGENESIS', 'KEGG_PENTOSE_PHOSPHATE_PATHWAY',
    'KEGG_CITRATE_CYCLE_TCA_CYCLE', 'KEGG_ONE_CARBON_POOL_BY_FOLATE', 'KEGG_PURINE_METABOLISM',
]
BREAST_FACTORS = ['HALLMARK_ESTROGEN_RESPONSE_EARLY', 'HALLMARK_ESTROGEN_RESPONSE_LATE',
                  'WP_BREAST_CANCER_PATHWAY']
VESSEL = {'VESSEL_PECAM1': ['PECAM1']}      # CD31 == PECAM1, a 1-gene vessel-marker factor

COLON_SIGNATURES = [
    'GOBP_INTESTINAL_EPITHELIAL_CELL_DIFFERENTIATION',
    'HALLMARK_TNFA_SIGNALING_VIA_NFKB', 'HALLMARK_INFLAMMATORY_RESPONSE',
    'HALLMARK_OXIDATIVE_PHOSPHORYLATION', 'HALLMARK_GLYCOLYSIS', 'HALLMARK_HYPOXIA',
    'HALLMARK_FATTY_ACID_METABOLISM', 'HALLMARK_MTORC1_SIGNALING',
    'HALLMARK_APICAL_JUNCTION', 'HALLMARK_APICAL_SURFACE', 'HALLMARK_BILE_ACID_METABOLISM',
    'HALLMARK_XENOBIOTIC_METABOLISM', 'GOBP_BICARBONATE_TRANSPORT', 'GOBP_SODIUM_ION_TRANSPORT',
    'GOBP_DEFENSE_RESPONSE_TO_BACTERIUM', 'GOBP_ANTIMICROBIAL_HUMORAL_RESPONSE',
]


@dataclass
class DatasetConfig:
    name: str                       # short key, also the cache subdir
    kind: str                       # 'cancer' | 'colon'
    adata_path: str
    cell_type_col: str              # obs column of cell-type annotation
    target_signatures: list         # predicted (pseudo-gene targets)
    neg_controls: list = field(default_factory=list)
    # cancer-only
    t_cell_types: list = field(default_factory=list)
    cancer_factors: list = field(default_factory=list)
    vessel: dict = field(default_factory=dict)
    manual_sets: dict = field(default_factory=dict)   # {name: [literal gene list]} targets
    # colon-only
    annot_col: str = None
    cell_type: str = None
    condition_col: str = None
    patient_col: str = None
    sample_col: str = None
    single_gene: str = None
    # regression settings
    method: str = 'l1'
    penalty: float = 0.005
    l1_ratio: float = 0.5
    standardize: bool = True
    n_sub: int = 50
    lower: int = 15
    upper: int = 85
    dbver: str = DBVER
    categories: tuple = CATEGORIES
    cache_root: str = CACHE_ROOT
    slurm_hint: str = ''            # human note for the dispatch notebook

    @property
    def cache_dir(self):
        return f'{self.cache_root}/{self.name}'


CONFIGS = {
    'cervical': DatasetConfig(
        name='cervical', kind='cancer',
        adata_path=f'{DATA_DIR}/Atera/FFPE_Human_Cervical_Cancer/LinearRegression/hvgall_x_adata.h5ad',
        cell_type_col='cell_type',
        t_cell_types=['Naive & Memory T Cells', 'Regulatory T Cells',
                      'Cytotoxic T Cells', 'Exhausted T Cells'],
        target_signatures=TCELL_TARGETS + METABOLIC_TARGETS, neg_controls=NEG_CONTROL,
        cancer_factors=CANCER_FACTORS, vessel=VESSEL, manual_sets=MANUAL_NEG_CONTROLS,
        slurm_hint='savio3 bigmem (large whole-transcriptome dataset)'),

    'breast': DatasetConfig(
        name='breast', kind='cancer',
        adata_path=f'{DATA_DIR}/Atera/FFPE_Human_Breast_Cancer/LinearRegression/hvgall_x_adata.h5ad',
        cell_type_col='cell_type',
        t_cell_types=['T Lymphocytes'],       # single T type -> one run per signature
        target_signatures=TCELL_TARGETS + METABOLIC_TARGETS, neg_controls=NEG_CONTROL,
        cancer_factors=BREAST_FACTORS + CANCER_FACTORS, vessel=VESSEL,
        manual_sets=MANUAL_NEG_CONTROLS,
        slurm_hint='savio3 htc'),

    'colon': DatasetConfig(
        name='colon', kind='colon',
        adata_path=f'{DATA_DIR}/Alexi_UC_Full_Data/processed_adata.h5ad',
        cell_type_col='25_06_11_ICI_5K_Coarse_annotations',
        annot_col='25_06_11_ICI_5K_Coarse_annotations', cell_type='IEC',
        condition_col='240927_Condition', patient_col='25_06_11_Patient_ID_HS', sample_col='slice',
        single_gene='DUOX2',
        target_signatures=COLON_SIGNATURES, neg_controls=NEG_CONTROL,
        slurm_hint='savio3 htc'),
}
