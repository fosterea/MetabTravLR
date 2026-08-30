

# Data Paths
# # Old data
# DATA_DIR = '/global/scratch/users/fosterangus/MetabTravLR/Data/Xenium'
DATASET  = 'Primary_Dermal_Melanoma'   # dataset folder under DATA_DIR

DATA_DIR = '/global/scratch/fsa/fc_wagnerlabfca/fosterangus/MetabTravLR/Data'
PROJECT = 'Xenium'
PROJECT_DATA_DIR = f'{DATA_DIR}/{PROJECT}'


# I looked over the initial SpaceTravLR results for the melanoma dataset and I think it would be helpful, as you suggested ,to curate a list of genes we're interested in. I think this would be a good starting point for validation - just testing for internal consistency and validate the method itself to see if we're capturing biological signal


# Negative control
# ACTB, B2M - housekeeping genes that should stay flat, not change too much between cell types (although maybe they wouldn't be part of a Xenium 5K panel...? Might need to double check that)

# Internal consistency check
# CD3D, CD3G alongisde CD3E - these are subunits of the same TCR-CD3 complex and should behave almost identically

# Positive control - general activation/metabolism that we may have some idea of how it works
# MYC, HIF1A - well-established glycolytic reprogramming TFs downstream of TCR/CD28. I'd expect glutamine transporters to move here

# Leaning towards discovery - exhaustion/differentiation trajectory
# TCF7, TOX, PDCD1, HAVCR2, LAG3 - actual trajectory instead of ENTPD1 standing in along for exhaustion



# And I've ranked this in what I think should be order of priority so it doesn't take too long to run.

# Oh that's unfortunate. Are you already running it? If not, you could check for GAPDH as well. 
# People also use it for housekeeping but it's also sorta relevant to general glycolysis metabolism. 
# Or TBP (some general transcription factor). It's ok if CD3D isn't there.

# Space travler focus gene sets
GENE_SETS = {
    'positive_v1': ['CD4', 'CD3E', 'IL2RA'],          # e.g. T-cell activity
    'negative_v1': ['CTLA4', 'FOXP3', 'IL10', 'ENTPD1'],  # e.g. exhaustion
    'negative_control': ['ACTB', 'B2M'],
    'consistency': ['CD3D', 'CD3G', 'CD3E'],
    'positive_control': ['MYC', 'HIF1A'],
    'discovery': ['TCF7', 'TOX', 'PDCD1', 'HAVCR2', 'LAG3'],
    'addition': ['GAPDH', 'TBP'],

    # UC related genes:
    'sodium_handlers': ['SLC9A3', 'SCNN1A', 'SCNN1G', 'SLC5A1', 'ATP1A1'],
    'downstream_outcomes': [
        'MAPK14', 'MAPK11', 'MAPK12', 'SGK1', # These get phosphorylated
        'RORC' # Th17 TF
    ]
}

FOCUS_GENES = list(set(dict.fromkeys(g for genes in GENE_SETS.values() for g in genes)))

