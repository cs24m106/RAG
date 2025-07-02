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

# add all the neccessary config params here that can be modified
config = {
    'embed_model': "all-minilm", # model used for embedding
    'chat_model' : "llama3.2", # model used for chat prompt
    'chunk_size' : 500, # chunk size to each retreival data
    'k_nearest'  : 5, # no. of relavant retreivals required
}

client = chromadb.PersistentClient(path=DB_PATH) # Configure ChromaDB client
llm = OllamaLLM(model=config["chat_model"]) # Configure Ollama Model use for chat prompt

# ------------ database setup ------------ 
class ChromaDBEmbeddingFunction:
    """Custom embedding function for ChromaDB using embeddings from Ollama"""
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        # Ensure the input is in a list format for processing
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)
    
    def name(self):
        return self.langchain_embeddings.model

# Initialize the embedding function with Ollama embeddings
OLLAMA_URL = "http://localhost:11434"  # Ollama local endpoint
embedFn = OllamaEmbeddings(model=config["embed_model"], base_url=OLLAMA_URL)

# Define a collection for the RAG workflow
collection_name = "kpi_rag_collection"
collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection of KPI documentations for RAG with Ollama"},
    embedding_function=ChromaDBEmbeddingFunction(embedFn)  # Use the custom embedding function
)

# ------------ pre-processing ------------ 
from src.input import read_docx
from pypdf import PdfReader

def read_pdf(file_path):
    """Read and extract text from a PDF file."""
    try:
        pdf = PdfReader(file_path)
        return '\n'.join(page.extract_text() for page in pdf.pages)
    except Exception as e:
        logger.error(f"Failed to read PDF file at {file_path}: {e}. Removing corrupted file.")
        os.remove(file_path) # comment to ignore
        return None

# Process the documents inside the directory given
def process_documents(doc_dir=DOCS_PATH):
    doc_ds = []
    for root, _, files in os.walk(doc_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            content = None
            if filename.lower().endswith('.docx'):
                content = read_docx(file_path)
            elif filename.lower().endswith('.pdf'):
                content = read_pdf(file_path)
            if content:
                data_dict = {"text": content, "source": filename}
                doc_ds.append(data_dict)
    return doc_ds

# Prepare Data to be added to chromaDB collection
from src.chunking import custom_text_splitter
doc_ds = process_documents()

doc_chunk_txt = []; doc_chunk_ids = []
for doc in doc_ds:
    chunks = custom_text_splitter(doc["text"], config["chunk_size"], 25, word_split = True)
    doc_chunk_txt.extend(chunks)
    doc_chunk_ids.extend([doc["source"]] * len(chunks))

# Add documents to the ChromaDB collection.
collection.add(documents=doc_chunk_txt, ids=doc_chunk_ids)

# ------------ RAG funtionalities ------------ 
from src.get_definitions import find_terms_and_abbreviations_in_sentence

def context_retrieval(query, n_results=5): # can pass multiple queries as list as well
    if type(query)==str:
        query = [query]
    return collection.query(query_texts=query, n_results=n_results, 
                            include=["metadatas", "documents", "distances", "embeddings"])

def submit_prompt(prompt:str):
    logger.info(f"LLM sumbitted with prompt:\n{prompt}")
    return llm.invoke(prompt)

# ToDo: add terms and abbreviations context in form of dict here
terms_definitions = {}
abbreviations_definitions = {}

def rag_pipeline(question:str, answer:str=None, should_enhance:bool=False):
    query = question
    if should_enhance:
        query = submit_prompt(f"Rephrase the question to be clear and concise:\n\'{question}\'")
        logger.info(f"llm enchanced query of the given question:\n{query}")
        
    formatted_terms, formatted_abbreviations = find_terms_and_abbreviations_in_sentence(terms_definitions, abbreviations_definitions, query)
    terms = ', '.join(formatted_terms)
    abbreviations = ', '.join(formatted_abbreviations)
    enhanced_query = f"{query}\n\nTerms and Definitions:\n{terms} \n\nAbbreviations:\n{abbreviations}"
    
    retrived_data = context_retrieval(enhanced_query, config["k_nearest"])
    context = []
    log_msg = []
    for i in range(len(retrived_data["documents"])): 
        cc =f"Retrieval {i+1}. This retrieval is performed from the document 3GPP '{retrived_data["ids"][i]}' : \n...{retrived_data["documents"][i]}...\n"
        mm = f"{i+1}. Source:{retrived_data["ids"][i]}, Embedding.size:{retrived_data["embeddings"][i].shape}, \nMetaData:{retrived_data["metadatas"][i]}, \nDocument:{retrived_data["documents"][i]}"
        context.append(cc)
        log_msg.append(mm)
    
    s_context = '\n'.join(context)
    s_log_msg = '\n'.join(log_msg)
    logger.info(f"Top[{config['k_nearest']}] Retrieved content: \n{s_log_msg}")

    augmented_prompt = f"""
Provide all the possible answers to the following question considering your knowledge and the text provided.
Question: {enhanced_query}
Considering the following context:
{s_context}
Provide all the possible answers to the following question considering your knowledge and the text provided.
Question: {question}
Ensure none of the answers provided contradicts your knowledge and each answer has at most 100 characters.
"""
    logger.info("######## Final Augmentation Prompt Completed ########")
    response = submit_prompt(augmented_prompt)
    # ToDo: measure similarity of embedding(response) and 
    return response, context


# main()
if __name__ == "__main__":
    # Example usage Define Question to query the RAG model
    question = "What is artificial intelligence?"  # 
    response = rag_pipeline(question)
    logger.info(f"######### ------ Response from LLM ------ #########\n{response}")