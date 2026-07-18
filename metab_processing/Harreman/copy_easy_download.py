import shutil
from pathlib import Path

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
    DATA_DIR='/global/scratch/users/fosterangus/MetabTravLR/Data/Xenium',
    SAVE_DIR='/global/home/users/fosterangus/Projects/MetabTravLR/SpaceTravLR/Results', 
    DATA_SET_NAME='Xenium_Tcell_Dataset'):

    copy_easy_download(DATA_DIR, f'{SAVE_DIR}/{DATA_SET_NAME}')

if __name__ == "__main__":
    save_easy_downloads()