import os
from preheader import CLONE_PATH, DOWN_PATH, RELEASE_VER
docs_dir = os.path.join(CLONE_PATH, "Documents")
down_dir = DOWN_PATH

from tqdm.auto import tqdm
from doc2docx import convert
from docx import Document
from src.storage import Storage
import sqlite3
from download_3gpp.options import UserOptions
from download_3gpp.download import Downloader
import shutil
import platform
import subprocess

import logging
logger = logging.getLogger(__name__) # Setup logging

def download_documents(release=None, series=None, standard=None, destination=DOWN_PATH, reset=False): # default: downloads all files
    if release is not None:
        destination = os.path.join(destination, f"Rel-{release}")
    if series is not None:
        destination = os.path.join(destination, f"{release}_series")
    # Update download directory
    global down_dir
    down_dir = destination
    if os.path.exists(destination) and not reset:
        logger.info(f"Download Dest-folder {destination} already exists => skipping downloading...")
        return destination
    
    # Step 1: Convert args into a list for UserOptions
    cmd_args = []
    if release is not None:
        cmd_args.extend(["--rel", str(release)])
    if series is not None:
        cmd_args.extend(["--series", str(series)])
    if standard is not None:
        cmd_args.extend(["--std", standard])

    # Step 2: Initialize UserOptions with default values and parsed args
    user_opts = UserOptions()
    user_opts.parse_arguments(cmd_args)
    user_opts.destination = destination

    # Step 3: Download 3GPP specs
    downloader = Downloader(user_opts)
    downloader.get_files()
    logger.info(f"Completed downloading files with config: rel:{user_opts.rel}, series:{user_opts.series}, std:{user_opts.std}")
    return destination

def extract_zipfiles(out_folder=docs_dir, inp_folder=down_dir, reset=False):
    # Step 4: Extract all docs into a single folder
    if os.path.exists(out_folder):
        if not reset:
            logger.info(f"Extracting Out-folder {out_folder} already exists => skipping extracting...")
            return
        shutil.rmtree(out_folder) # delete existing documents dir (if flaged with hard reset)
    os.makedirs(out_folder, exist_ok=True)
    if not os.path.exists(inp_folder) or not os.path.isdir(inp_folder):
        logger.warning(f"Input folder {inp_folder} is invalid. Resetting to {DOWN_PATH}.")
        inp_folder = DOWN_PATH

    for root, _, files in os.walk(inp_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(('.doc', '.docx')):
                shutil.copy2(file_path, out_folder)
            elif file.lower().endswith('.zip'):
                try:
                    shutil.unpack_archive(file_path, out_folder)
                except Exception as e:
                    logger.warning(f"Failed to extract {file_path}: {e}")

def read_docx(file_path):
    """Read and extract text from a DOCX file."""
    try:
        doc = Document(file_path)
        return '\n'.join(para.text for para in doc.paragraphs)
    except Exception as e:
        logger.error(f"Failed to read DOCX file at {file_path}: {e}. Removing corrupted file.")
        os.remove(file_path) # comment to ignore
        return None

def get_documents(series_list, folder_path=docs_dir, storage_name='Documents.db', dataset_name='Standard', hard_reset=False):
    """Retrieve and process documents from a folder, storing them in a database if not already present."""
    db_path = os.path.join(CLONE_PATH, storage_name)
    if os.path.exists(db_path):
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA schema_version;")
        except sqlite3.DatabaseError:
            logger.warning(f"{db_path} is not a valid database. Removing corrupted file.")
            os.remove(db_path) # comment to ignore
    else:
        logger.warning(f"Database path: \"{db_path}\"  does not exist")
    
    storage = Storage(db_path)
    storage.create_dataset(dataset_name)
    document_ds = []
    file_list = []

    # Download and extract Documents (set reset flag carefully, will take time)
    download_documents(release=RELEASE_VER, reset=hard_reset) # debug for release 18 only for now
    extract_zipfiles(out_folder=folder_path, reset=hard_reset)

    # Check and convert .doc files to .docx
    convert_docs_to_docx(folder_path, clear_space=True)

    # Prepare list of .docx files for processing
    file_list = [f for f in os.listdir(folder_path) if valid_file(f, series_list)]

    # Process each document
    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        process_document(file_path, filename, storage, document_ds, dataset_name)

    storage.close()
    return document_ds

def convert_docs_to_docx(folder_path, clear_space=False):
    """Convert .doc files in a folder to .docx format if any."""
    has_doc = any(f.endswith('.doc') for f in os.listdir(folder_path))
    if not has_doc:
        return
    if platform.system().lower() == "linux":
        # Convert all .doc files in the folder to .docx using LibreOffice
        doc_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.doc') and not f.endswith('.docx')]
        try:
            subprocess.run(
                ["libreoffice", "--headless", "--convert-to", "docx", "--outdir", folder_path] + doc_files,
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            logger.warning(f"Failed to convert DOC files using LibreOffice: {e}")
    else:
        # Use doc2docx's convert function for Windows or other OS
        try:
            convert(folder_path)
        except Exception as e:
            logger.warning(f"Failed to convert DOC files using doc2docx: {e}")

    if not clear_space:
        return
    # clear any other temp files/folders other than .docx files
    for f in os.listdir(folder_path):
        file_path = os.path.join(folder_path, f)
        if f.endswith('.docx'):
            continue
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.warning(f"Failed to delete {file_path}: {e}")

def valid_file(filename, series_list):
    """Check if a file should be processed based on its name and series list."""
    return filename.endswith(".docx") and not filename.startswith("~$") and (not filename[:2].isnumeric() or int(filename[:2]) in series_list)

def process_document(file_path, filename, storage, document_ds, dataset_name):
    """Process a single document file."""
    if storage.is_id_in_dataset(dataset_name, filename):
        data_dict = storage.get_dict_by_id(dataset_name, filename)
        document_ds.append(data_dict)
    else:
        content = read_docx(file_path)
        if content:
            data_dict = {'id': filename, 'text': content, 'source': filename}
            # id (unique search idx) -> filename, since it is what being used for searching the entry
            document_ds.append(data_dict)
            storage.insert_dict(dataset_name, data_dict)
