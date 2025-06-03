import os
import shutil
from ftplib import FTP
from zipfile import ZipFile
from datetime import datetime
import pytz
from download_3gpp import download_3gpp
from docx import Document

# Import the delete_sections function from remove_content.py
from preprocess.remove_content import delete_sections

# Define directories
DOWNLOAD_DIR = 'downloads'
EXTRACT_DIR = 'extracted_docs'
KNOWLEDGE_BASE_DIR = 'knowledge_base'

# Create directories if they don't exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

def get_ftp_file_time(ftp, filename):
    """Get the modification time of a file on the FTP server."""
    try:
        # Use MDTM command to get the modification time
        timestamp = ftp.sendcmd(f"MDTM {filename}")
        # Response format: '213 YYYYMMDDHHMMSS'
        time_str = timestamp[4:]
        return datetime.strptime(time_str, '%Y%m%d%H%M%S').replace(tzinfo=pytz.UTC)
    except Exception:
        return None

def needs_update(local_path, ftp, ftp_path):
    """Check if the local file needs to be updated based on FTP timestamp."""
    if not os.path.exists(local_path):
        return True
    local_mtime = datetime.fromtimestamp(os.path.getmtime(local_path), tz=pytz.UTC)
    ftp_mtime = get_ftp_file_time(ftp, ftp_path)
    return ftp_mtime is None or ftp_mtime > local_mtime

def download_and_extract():
    """Download the latest 3GPP documents and extract them."""
    # Connect to the FTP server
    ftp = FTP('ftp.3gpp.org')
    ftp.login()  # Anonymous login
    ftp.cwd('/Specs/latest')

    # Download only updated files
    download_3gpp.download_latest_specs(download_dir=DOWNLOAD_DIR, ftp=ftp, needs_update=needs_update)

    # Extract the downloaded zip files
    for root, _, files in os.walk(DOWNLOAD_DIR):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                with ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(EXTRACT_DIR)
    
    ftp.quit()

def preprocess_documents():
    """Preprocess the extracted .doc files and save them to knowledge_base."""
    for root, _, files in os.walk(EXTRACT_DIR):
        for file in files:
            if file.endswith('.doc'):
                doc_path = os.path.join(root, file)
                output_path = os.path.join(KNOWLEDGE_BASE_DIR, file)
                try:
                    delete_sections(doc_path, output_path)
                except Exception as e:
                    print(f"Error preprocessing {file}: {e}")

def main():
    """Main function to orchestrate the download, extraction, and preprocessing."""
    print("Starting the download and preprocessing of 3GPP documents...")
    download_and_extract()
    preprocess_documents()
    print("Process completed successfully. Cleaned documents are in the 'knowledge_base' directory.")

if __name__ == "__main__":
    main()