import shutil
import sys
from pathlib import Path

# Make the repo root importable regardless of CWD or machine: walk up from this
# file until we hit a repo marker, then put that dir on sys.path.
_root = next(
    (p for p in Path(__file__).resolve().parents
     if (p / ".git").exists() or (p / "setup.py").exists()),
    Path(__file__).resolve().parent,
)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from metab_processing.metab_travlr_config import PROJECT_DATA_DIR

def copy_easy_download(src_dir, dest_dir):
    src = Path(src_dir)
    dest = Path(dest_dir)
    
    for subfolder in src.iterdir():
        if subfolder.is_dir():
            easy_down = subfolder / "easy_download"
            if easy_down.exists():
                target = dest / subfolder.name / "easy_download"
                shutil.copytree(easy_down, target, dirs_exist_ok=True)

def save_easy_downloads(
    DATA_DIR=PROJECT_DATA_DIR,
    SAVE_DIR='/global/home/users/fosterangus/Projects/MetabTravLR/SpaceTravLR/Results', 
    DATA_SET_NAME='Xenium_Tcell_Dataset'):

    copy_easy_download(DATA_DIR, f'{SAVE_DIR}/{DATA_SET_NAME}')

if __name__ == "__main__":
    save_easy_downloads()