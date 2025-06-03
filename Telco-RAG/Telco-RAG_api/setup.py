import os
import git
from git import RemoteProgress
from tqdm import tqdm
import shutil

class CloneProgress(RemoteProgress):
    def __init__(self):
        super().__init__()
        self.pbar = tqdm()

    def update(self, op_code, cur_count, max_count=None, message=''):
        self.pbar.total = max_count+1
        self.pbar.n = cur_count
        self.pbar.refresh()

folder_url = "https://huggingface.co/datasets/netop/Embeddings3GPP-R18"
clone_directory = "./3GPP-Release18"

if not (os.path.exists(clone_directory) and "Embeddings" in os.listdir(clone_directory)):
    print("Downloading 3GPP dataset for RAG...")
    git.Repo.clone_from(folder_url, clone_directory, progress=CloneProgress())
    # Pull LFS files
    print("Pulling LFS files...")
    repo = git.Repo(clone_directory)
    repo.git.lfs('pull')
    print("LFS files pulled successfully!")
else:
    print("Folder already exists. Skipping cloning.")