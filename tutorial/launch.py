import sys
import scanpy as sc
sys.path.append('../src')

from SpaceTravLR.spaceship import SpaceShip

spacetravlr = SpaceShip(
    name='myTonsil', 
    outdir='/global/scratch/users/fosterangus/MetabTravLR/Data/SpaceTravlrTonsilTest'
)
assert spacetravlr.is_everything_ok()

spacetravlr.fit()
