# ------------ setup environment ------------ 
import os, sys
VENV_PATH = os.environ.get('VIRTUAL_ENV') # virtual environment path
curr_path = os.path.abspath(__file__)
REPO_DIR = "RAG"
REPO_PATH = curr_path[:curr_path.find(REPO_DIR) + len(REPO_DIR)]

ROOT_DIR = REPO_DIR + "_api"
ROOT_PATH = os.path.join(REPO_PATH, "ref/Telco-RAG/Telco-RAG_api")
sys.path.append(ROOT_PATH)
DOCS_PATH = os.path.join(os.getcwd(), "documents")
DB_PATH = os.path.join(os.getcwd(), "chroma_db")

# libraries
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb, logging
import preheader # import for custom logger

# ------------ configurations ------------ 
logger = logging.getLogger(__name__) # Setup logging
# Supress http conn logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
# Suppress ChromaDB telemetry and component logs
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# add all the neccessary config params here that can be modified
config = {
    'embed_model'   : "all-minilm", # model used for embedding
    'chat_model'    : "llama3.2", # model used for chat prompt
    'chunk_size'    : 500, # chunk size to each retreival data
    'chunk_overlap' : 25, # no.of char allowed to overlap
    'k_nearest'     : (1,3), # no. of relavant retreivals required, idx: 0 => from xcels, 1 => from docs
    'generate'      : False, # to use llm to generate response or not
}

# Configure ChromaDB client
import chromadb

# The default Chroma client is ephemeral, meaning it will not save to disk
#client = chromadb.Client()

# local copy of database on disk
#client = chromadb.PersistentClient(path=DB_PATH)

# run in a seperate terminal: ```chroma run --host 127.0.0.1 -port 8000 --path /content/chroma_db &``` 
# runs chromadb in the host using exist database in given dir [& => runs in background] [port 8000 by default]
client = chromadb.HttpClient(host='127.0.0.1', port=8000)

llm = OllamaLLM(model=config["chat_model"]) # Configure Ollama Model use for chat prompt

# ------------ database setup ------------ 
import numpy as np
class ChromaDBEmbeddingFunction:
    """Custom embedding function for ChromaDB using embeddings from Ollama"""
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        # Ensure input is always a list
        if isinstance(input, str):
            input = [input]

        # Get embeddings from Ollama (returns list of lists)
        embeddings = self.langchain_embeddings.embed_documents(input)

        # Convert each list-based embedding to a NumPy array
        return [np.array(embedding) for embedding in embeddings]

    def name(self):
        return self.langchain_embeddings.model

# Initialize the embedding function with Ollama embeddings
OLLAMA_URL = "http://localhost:11434"  # Ollama local endpoint
embedFn = OllamaEmbeddings(model=config["embed_model"], base_url=OLLAMA_URL)

# Define a collection for the RAG workflow
chromaEmbedFn = ChromaDBEmbeddingFunction(embedFn)
doc_collection_name = "KPI_and_PM_lte"
doc_collection = client.get_or_create_collection(
    name=doc_collection_name,
    metadata={"description": "A collection of KPI & PM documentations."},
    embedding_function=chromaEmbedFn  # Use the custom embedding function
)
csv_collection_name = "KPI_and_PM_formulas"
csv_collection = client.get_or_create_collection(
    name=csv_collection_name,
    metadata={"description": "A collection of KPI & PM Database Formulas."},
    embedding_function=chromaEmbedFn  # Use the custom embedding function
)

collections = [csv_collection, doc_collection] # order as per pipeline
assert len(collections) == len(config["k_nearest"]), "Mismatch: collections and config['k_nearest'] must have the same length"

# ------------ pre-processing ------------ 
from docx import Document
from pypdf import PdfReader
import pandas as pd
from openpyxl import load_workbook

def read_docx(file_path):
    """Read and extract text from a DOCX file."""
    text_extract = None; metadata = None
    try:
        doc = Document(file_path)
        text_extract = '\n'.join(para.text for para in doc.paragraphs)
        
    except Exception as e:
        logger.error(f"Failed to read DOCX file at {file_path}: {e}. Removing corrupted file.")
        os.remove(file_path) # comment to ignore
    try:
        prop = doc.core_properties # remove private or unnecessary attrb startin with '_'
        metadata = {d: getattr(prop, d) for d in dir(prop) if not d.startswith('_')}
    except Exception as e:
        logger.error(f"Failed to read DOCX file metadata! {file_path}: {e}")
        
    return text_extract, metadata

def read_pdf(file_path):
    """Read and extract text from a PDF file."""
    text_extract = None; metadata = None
    try:
        pdf = PdfReader(file_path)
        text_extract = '\n'.join(page.extract_text() for page in pdf.pages)
    except Exception as e:
        logger.error(f"Failed to read PDF file at {file_path}: {e}. Removing corrupted file.")
        os.remove(file_path) # comment to ignore
    try:
        prop = pdf.metadata #strip the leading '/' from all string keys in the metadata.
        metadata = {key.lstrip('/') if isinstance(key, str) else key: value for key, value in prop.items()}
    except Exception as e:
        logger.error(f"Failed to read DOCX file metadata! {file_path}: {e}")
    return text_extract, metadata

def read_xcel(file_path):
    """Read and extract list of texts from a XLSX or CSV file."""
    text_extract = None; metadata = None
    try:
        # Determine file type
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Capture original DataFrame properties
        original_num_columns = len(df.columns)
        original_missing = df.isna().sum().to_dict()
        all_na_cols = [col for col in df.columns if df[col].isna().all()]
        num_dropped_columns = len(all_na_cols)

        # Drop columns with all missing values
        df = df.dropna(axis=1, how='all')
        # Process each row to generate enriched strings
        text_extract = []
        for _, row in df.iterrows():
            attributes = []
            for col, val in row.items():
                if pd.notna(val) and val != '':  # Only include valid values
                    attributes.append(f"{col}: {val}")
            if attributes:
                text_extract.append('\n'.join(attributes))

        # Add DataFrame properties (always present)
        metadata = {
            'original_num_columns': original_num_columns,
            'num_dropped_columns': num_dropped_columns,
            'dropped_columns': all_na_cols,
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage': int(df.memory_usage(deep=True).sum()),
            'missing_values_before_drop': original_missing,
        }
    except Exception as e:
        logger.error(f"Failed to read XLSX/CSV file at {file_path}: {e}. Removing corrupted file.")
        #os.remove(file_path) # comment to ignore
    
    try:
        if file_path.endswith('.xlsx'):     
            wb = load_workbook(file_path, read_only=True, keep_vba=False)
            props = wb.properties
            # Extract relevant Excel properties with 'xlsx_' prefix
            file_meta = {f'xlsx_{k}': (v.isoformat() if hasattr(v, 'isoformat') else v)
                        for k, v in vars(props).items()
                        if not k.startswith('_') and v}
            wb.close()
            metadata.update(file_meta)
        else:
            file_stats = os.stat(file_path)
            file_meta = {
            'file_name': os.path.basename(file_path),
            'file_size': file_stats.st_size,
            'modified_time': file_stats.st_mtime,
            }
            metadata.update(file_meta)
    except Exception as wb_error:
        logger.error(f"Failed to read XLSX/CSV file metadata! {file_path}: {e}")

    # 3. text_extract = list of enriched string of each row, return the metadata of the file as well
    return text_extract, metadata

# Process the documents inside the directory given
def process_documents(doc_dir=DOCS_PATH):
    doc_ds = []
    for root, _, files in os.walk(doc_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            content = None; meta = None
            
            ext = os.path.splitext(filename)[1].lower() # process each file based on extension
            if ext == '.docx':
                content, meta = read_docx(file_path)
            elif ext == '.pdf':
                content, meta = read_pdf(file_path)
            if ext in ('.xlsx', '.csv'):
                content, meta = read_xcel(file_path)
            
            if meta: # conv all non primitive data types inside meta to string
                meta = {k: (str(v) if not isinstance(v, (str, int, float, bool)) else v) for k, v in meta.items()}
            if content: # create data_dict
                data_dict = {"text": content, "source": filename, "metadata": meta}
                doc_ds.append(data_dict)
    return doc_ds

import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Prepare Data to be added to chromaDB collection
def prepare_dataset():
    doc_ds = process_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=[
            "([\s,.\-!?\[\]\(\){}\":;<>]+)",  # Use regex group for complex separators
            " ", "\n", "\t", "\r", "\f", "\v" # Mimics word-splitting
            ],
        length_function=len,
    )
    doc_chnk_txt = []; doc_chnk_ids = []; doc_chnk_mta = []

    for doc in doc_ds:
        chunks = []
        ext = os.path.splitext(doc['source'])[1].lower()
        
        if ext in ('.xlsx', '.csv'): # chunks already done for xcel type files
            chunks = doc["text"]
            ids = [f"{doc['source']}_entry{i+1}" for i in range(len(chunks))]
            csv_collection.upsert(ids=ids, documents=chunks, metadatas=[doc["metadata"]]*len(chunks)) # Add to database collection
        
        else:
            txt = doc["text"] # Apply some additional pre-processing
            sub = re.sub(r'[ \t\r\f\v]*\n[ \t\r\f\v]*', '\n', txt)
            clean_txt = re.sub(r'\n+', '\n', sub).strip()
            chunks = text_splitter.split_text(clean_txt) # Split text using LangChain's splitter
            ids = [f"{doc['source']}_chunk{i+1}" for i in range(len(chunks))]
            doc_collection.upsert(ids=ids, documents=chunks, metadatas=[doc["metadata"]]*len(chunks)) # Add to documentation collection
        
        logger.info(f"No. of chunks extracted out of f{doc['source']} = {len(chunks)}.\nMetadata: {doc['metadata']}")

# ------------ RAG funtionalities ------------ 
from src.get_definitions import find_terms_and_abbreviations_in_sentence

def context_retrieval(query, collection, n_results=5): # can pass multiple queries as list as well
    if type(query)==str:
        query = [query]
    retrieval = collection.query(query_texts=query, n_results=n_results, 
                            include=["metadatas", "documents", "distances", "embeddings"])
    #  rearrange result format of each key
    results = []
    for i in range(len(query)):
        results.append({})
        for key in retrieval:
            if retrieval[key] is not None:
                results[-1][key] = retrieval[key][i]
    if len(query) == 1: # flatten nested list result
        results = results[0]
    return results # rearrange format

def submit_prompt(prompt:str):
    if config["generate"] == False:
        logger.warning("config.generate is set to false, cannot use llm!")
        return prompt
    logger.info(f"LLM sumbitted with prompt:\n{prompt}")
    return llm.invoke(prompt)

def compare_embeddings(embed1, embed2):
    # Convert embeddings to numpy arrays for dot product calculation
    embedding1_array = np.array(embed1)
    embedding2_array = np.array(embed2)

    # Calculate and return the dot product
    dot_product = np.dot(embedding1_array, embedding2_array)
    return dot_product

def compare_str(embedFn, inp1, inp2):
    """
    Returns the dot product (cosine similarity) of embeddings for two given text strings.

    The dot product value ranges from -1 to 1:
        - Values close to 1: high similarity.
        - Values close to -1: high dissimilarity.
        - Values close to 0: no apparent similarity.

    Parameters:
        embeddings_model: Embeddings model used to generate embeddings.
        inp1: First text string or embedding vector
        inp2: Second text string or embedding vector

    Returns:
        float: Dot product of the embeddings for text1 and text2.
    """
    e1 = inp1; e2 = inp2

    # Get the embeddings for the two text strings
    if (isinstance(inp1, str)):
        e1 = embedFn.embed_query(inp1)
    if (isinstance(inp1, str)):
        e2 = embedFn.embed_query(inp1)

    return compare_embeddings(e1, e2)

# ToDo: add terms and abbreviations context in form of dict here
terms_definitions = {}
abbreviations_definitions = {}

def rag_pipeline(question:str, retrieve_only:bool=False, should_enhance:bool=False):
    query = question

    if not retrieve_only:
        if should_enhance:
            query = submit_prompt(f"Rephrase the question to be clear and concise:\n\'{question}\'")
        
        if terms_definitions != {} or abbreviations_definitions != {}:
            formatted_terms, formatted_abbreviations = find_terms_and_abbreviations_in_sentence(terms_definitions, abbreviations_definitions, query)
            terms = ', '.join(formatted_terms)
            abbreviations = ', '.join(formatted_abbreviations)
        if terms_definitions != {}:
            query = f"{query}\n\nTerms and Definitions:\n{terms}"
        if abbreviations_definitions != {}:
            query = f"{query}\n\nAbbreviations:\n{abbreviations}\n"
        logger.info(f"Final Enhanced version: [llm:{should_enhance} + Terms & Abbreviations] Query of the given question:\n{query}")
    
    context = []; log_msg = []; qq = query
    for j in range(len(collections)):
        retrived_data = context_retrieval(qq, collections[j], config["k_nearest"][j])
        query_embedding = embedFn.embed_query(qq)
        #logger.warning(retrived_data) # comment it

        for i in range(len(retrived_data["documents"])): 
            cc =f"\nRetrieval {j+1}.{i+1}. This retrieval is performed from the document '{retrived_data["ids"][i]}' : \n...{retrived_data["documents"][i]}...\n"
            retrieval_embedding = retrived_data["embeddings"][i]
            mm = f"{j+1}.{i+1}. Source:{retrived_data["ids"][i]}, Distance:{retrived_data["distances"][i]}, Similarity.score:{compare_embeddings(query_embedding, retrieval_embedding)}, " + \
            f"Embedding.size:{retrieval_embedding.shape}, \nMetaData:{repr(retrived_data["metadatas"][i])}, \n>>> Document:\n{retrived_data["documents"][i]}\n"
            context.append(cc)
            log_msg.append(mm)
        
        qq = qq + "\n".join(retrived_data["documents"]) # keep unnecessary content out when pipelining
    
    s_context = '\n'.join(context)
    s_log_msg = '\n'.join(log_msg)
    logger.info(f"Top[{config['k_nearest']}] Retrieved content: \n{s_log_msg}")

    response = None
    if not retrieve_only:
        augmented_prompt = f"""Query:\n{query}\nContext:\n{s_context}
>>> Provide all the possible answers to the following question considering your knowledge and the text provided.
Question: {question}
>>> Ensure none of the answers provided contradicts your knowledge.
"""
        logger.info("######## Final Augmentation Prompt Completed ########")
        response = submit_prompt(augmented_prompt)
    
    return response, context


# main() -> conc only on retreival
if __name__ == "__main__":
    prepare_dataset()

    # Example usage: Define Question to query the RAG model
    questions = [
        "Describe active E-RABs in a few lines.", 
        "What are the parameters related to QoS parameter?",
        "What are handover related measurements?",
        "What is IP Throughput?",
    ]
    answers = [] # add answers to measure similarity

    for i in range(len(questions)):
        logger.debug("\n\n")
        logger.info(f">>> Question: {questions[i]}")
        response, context = rag_pipeline(questions[i], retrieve_only= (not config['generate']), should_enhance=False) # change config
        if response != None:
            logger.info(f"######### ------ Response from LLM Generated ------ #########\n{response}")
            if 0 <= i < len(answers):
                logger.info(f">>> Expected Answer [similarity score w.r.t llm response = {compare_str(answers[i], response)}]: {answers[i]}")
        input("PAUSED! Press <ENTER> to continue...")
    