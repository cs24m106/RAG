import os
import argparse
import zipfile
import subprocess
from tqdm import tqdm
from doc2docx import convert
from download_3gpp.options import UserOptions
from download_3gpp.download import Downloader
from preprocess.remove_content import delete_sections
import platform

# Define paths
DOWNLOAD_DIR = "data/downloads"
EXTRACT_DIR = "data/extracted_docs"
OUTPUT_DIR = "data/knowledge_base"

# Create directories if they don't exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_zipfiles(input_dir, output_dir):
    """Extract each .zip file preserving directory structure."""
    zip_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.zip'):
                zip_files.append((root, file))

    for root, file in tqdm(zip_files, desc="Extracting", unit=" zip"):
        zip_path = os.path.join(root, file)
        extract_to = os.path.join(output_dir, os.path.relpath(root, input_dir))
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        # print(f"Extracted {zip_path} to {extract_to}")
    #print(f"Extracting {len(zip_files)} \".zip\" files from {input_dir} to {output_dir} completed")
    

def preprocess_files(input_dir, output_dir):
    """Run preprocessing on each .doc/x file using remove_content.py."""
    docx_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.doc') or file.endswith('.docx'):
                docx_files.append((root, file))

    for root, file in tqdm(docx_files, desc="Preprocessing", unit=" file"):
        input_path = os.path.join(root, file)
        
        # Handle .doc files by converting them to .docx
        # ===
        if file.endswith(".doc"):            
            try:
                input_path = convert_doc_to_docx(root, file)
            except subprocess.CalledProcessError as e:
                print(f"Failed to convert {input_path}: {e}")
                continue
        # ===

        output_path = os.path.join(output_dir, os.path.relpath(input_path, input_dir))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            delete_sections(input_path, output_path)
            # print(f"Cleaned {input_path} -> {output_path}")
        except Exception as e:
            print(f"Failed to process {input_path}: {e}")


def convert_doc_to_docx(root, file): # convert in same directory
    """Convert .doc file to .docx using LibreOffice or doc2docx."""
    input_path = os.path.join(root, file)
    output_path = os.path.join(root, file.replace(".doc", ".docx"))

    if platform.system().lower() == "linux":
        subprocess.run([
            "libreoffice",
            "--headless",
            "--convert-to", "docx",
            "--outdir", root,
            input_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        # Use doc2docx's convert function for Windows or other OS
        convert(input_path, output_path)
    #print(f"Preprocessing {len(docx_files)} \".docx\" files from {input_dir} to {output_dir} completed")
    return output_path

def main():
    # Step 1: Parse command-line arguments for user-visible options
    parser = argparse.ArgumentParser(description="Acquire 3GPP standards packages from archive")
    parser.add_argument("--rel", type=int, help="3GPP release number to target, default 'all'")
    parser.add_argument("--series", type=int, help="3GPP series number to target, default 'all'")
    parser.add_argument("--std", type=str, help="3GPP standard number to target, default 'all'")
    args = parser.parse_args()

    # Step 2: Convert parsed args into a list for UserOptions
    cmd_args = []
    if args.rel is not None:
        cmd_args.extend(["--rel", str(args.rel)])
    if args.series is not None:
        cmd_args.extend(["--series", str(args.series)])
    if args.std is not None:
        cmd_args.extend(["--std", args.std])

    # Step 3: Initialize UserOptions with default values and parsed args
    user_options = UserOptions()
    user_options.parse_arguments(cmd_args)
    user_options.destination = DOWNLOAD_DIR
    
    # Step 4: Download 3GPP specs
    downloader = Downloader(user_options)
    downloader.get_files()
    print(f"Completed downloading files with config: rel:{user_options.rel}, series:{user_options.series}, std:{user_options.std}")

    # Step 5: Extract .zip files
    extract_zipfiles(DOWNLOAD_DIR, EXTRACT_DIR)

    # Step 6: Preprocess extracted .docx files
    preprocess_files(EXTRACT_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()