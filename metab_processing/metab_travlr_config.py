

# Data Paths
# # Old data
# DATA_DIR = '/global/scratch/users/fosterangus/MetabTravLR/Data/Xenium'
DATASET  = 'Primary_Dermal_Melanoma'   # dataset folder under DATA_DIR

DATA_DIR = '/global/scratch/fsa/fc_wagnerlabfca/fosterangus/MetabTravLR/Data'
PROJECT = 'Xenium'
PROJECT_DATA_DIR = f'{DATA_DIR}/{PROJECT}'


# Space travler focus gene sets
GENE_SETS = {
    'positive': ['CD4', 'CD3E', 'IL2RA'],          # e.g. T-cell activity
    'negative': ['CTLA4', 'FOXP3', 'IL10', 'ENTPD1'],  # e.g. exhaustion
}

FOCUS_GENES = list(dict.fromkeys(g for genes in GENE_SETS.values() for g in genes))

