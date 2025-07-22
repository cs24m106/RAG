# ------------ setup environment ------------ 
import os, sys
sys.path.append("..")
curr_dir = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(curr_dir, "documents")
DB_PATH = os.path.join(curr_dir, "chroma_db")

# libraries
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import numpy as np
import chromadb, logging, re
import preheader # import for custom logger

# ------------ configurations ------------ 
logger = logging.getLogger(__name__) # Setup logging
# Supress http conn logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
# Suppress ChromaDB telemetry and component logs
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

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

# ------------ pre-processing ------------
from preprocess import process_documents, find_terms_and_abbreviations_in_sentence
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

    for doc in doc_ds:
        chunks = []
        ext = os.path.splitext(doc['source'])[1].lower()
        
        if ext in ('.xlsx', '.csv'): # chunks already done for xcel type files
            chunks = doc["text"]
            ids = [f"{doc['source']}_entry{i+1}" for i in range(len(chunks))]
            if 'formula' in doc['source'].lower():
                formula_collection.upsert(ids=ids, documents=chunks, metadatas=[doc["metadata"]]*len(chunks))
            else:
                csv_collection.upsert(ids=ids, documents=chunks, metadatas=[doc["metadata"]]*len(chunks))
        
        else:
            txt = doc["text"] # Apply some additional pre-processing
            sub = re.sub(r'[ \t\r\f\v]*\n[ \t\r\f\v]*', '\n', txt)
            clean_txt = re.sub(r'\n+', '\n', sub).strip()
            chunks = text_splitter.split_text(clean_txt) # Split text using LangChain's splitter
            ids = [f"{doc['source']}_chunk{i+1}" for i in range(len(chunks))]
            if 'formula' in doc['source'].lower():
                formula_collection.upsert(ids=ids, documents=chunks, metadatas=[doc["metadata"]]*len(chunks))
            else:
                doc_collection.upsert(ids=ids, documents=chunks, metadatas=[doc["metadata"]]*len(chunks))
        
        logger.info(f"No. of chunks extracted out of {doc['source']} = {len(chunks)}.\nMetadata: {doc['metadata']}")

# ------------ RAG funtionalities ------------ 
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


def rag_pipeline(question:str, retrieve_only:bool=False, should_enhance:bool=False):
    query = question

    if not retrieve_only:
        if should_enhance:
            query = submit_prompt(f"Rephrase the question to be clear and concise:\n\'{question}\'")
        
        if config["terms"] != {} or config["abbreviations"] != {}:
            formatted_terms, formatted_abbreviations = find_terms_and_abbreviations_in_sentence(config["terms"], config["abbreviations"], query)
            terms = ', '.join(formatted_terms)
            abbreviations = ', '.join(formatted_abbreviations)
        if config["terms"] != {}:
            query = f"{query}\n\nTerms and Definitions:\n{terms}"
        if config["abbreviations"] != {}:
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


# ------------ Functional Modularity ------------ 
def initialize_database(
        embed_model="all-minilm", 
        chat_model="llama3.2", 
        chunk_size=500, 
        chunk_overlap=25, 
        k_nearest=(1,1,3), 
        terms_definitions={}, 
        abbreviations_definitions={},
        refresh=False):
    """
    Parameters:
        embed_model: Model used for embedding.
        chat_model: Model used for chat prompt.
        chunk_size: Chunk size for each retrieval data.
        chunk_overlap: Number of characters allowed to overlap.
        k_nearest: Number of relevant retrievals required; index (0 => from formulas, 1 => from Excels, 2 => from docs)
        rag_generate: Whether to use LLM to generate RAG-based query.
        sql_generate: Whether to use LLM to generate SQL-based query.
        terms_definitions: Dictionary of terms and their definitions.
        abbreviations_definitions: Dictionary of abbreviations and their definitions.
        refresh: Reset chroma database if documents are modified
    """
    
    global config, client, llm, embedFn, collections, doc_collection, formula_collection, csv_collection # declare on global scope
    # add all the neccessary config params here that can be modified
    config = {
        'embed_model'   : embed_model, 
        'chat_model'    : chat_model,
        'chunk_size'    : chunk_size,
        'chunk_overlap' : chunk_overlap,
        'k_nearest'     : k_nearest,
        'rag_generate'  : False,
        'sql_generate'  : True,
        # TO DO: for latter
        'terms'         : terms_definitions,
        'abbreviations' : abbreviations_definitions,
    }

# Configure ChromaDB client
    # The default Chroma client is ephemeral, meaning it will not save to disk
    #client = chromadb.Client()

    # local copy of database on disk
    #client = chromadb.PersistentClient(path=DB_PATH)

    # run in a seperate terminal: ```chroma run --host 127.0.0.1 --port 8000 --path /content/chroma_db &``` 
    # runs chromadb in the host using exist database in given dir [& => runs in background] [port 8000 by default]
    client = chromadb.HttpClient(host='127.0.0.1', port=8000)

    llm = OllamaLLM(model=config["chat_model"]) # Configure Ollama Model use for chat prompt

    embedFn = OllamaEmbeddings(model=config["embed_model"], base_url=OLLAMA_URL)
    chromaEmbedFn = ChromaDBEmbeddingFunction(embedFn)

# ------------ database setup ------------ 
    # Delete collections if they exist
    if refresh:
        for cname in ["KPI_and_PM_lte", "KPI_and_PM_formulas", "KPI_and_PM_analytics"]:
            try:
                client.delete_collection(name=cname)
            except Exception:
                pass
    
    # Define a collection for the RAG workflow
    doc_collection = client.get_or_create_collection(
        name="KPI_and_PM_lte",
        metadata={"description": "A collection of KPI & PM documentations."},
        embedding_function=chromaEmbedFn  # Use the custom embedding function
    )
    formula_collection = client.get_or_create_collection(
        name="KPI_and_PM_formulas",
        metadata={"description": "A collection of KPI & PM Database Formulas."},
        embedding_function=chromaEmbedFn  # Use the custom embedding function
    )
    csv_collection = client.get_or_create_collection(
        name="KPI_and_PM_analytics",
        metadata={"description": "A collection of KPI & PM Database analysis like coorelation."},
        embedding_function=chromaEmbedFn  # Use the custom embedding function
    )

    collections = [formula_collection, csv_collection, doc_collection] # order as per pipeline
    assert len(collections) == len(config["k_nearest"]), "Mismatch: collections and config['k_nearest'] must have the same length"
    return config


# main() -> conc only on retreival
if __name__ == "__main__":
    config = initialize_database()
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
        pure_retrieval = (not config["rag_generate"])
        response, context = rag_pipeline(questions[i], retrieve_only=pure_retrieval, should_enhance=pure_retrieval) # change config
        if response != None:
            logger.info(f"######### ------ Response from LLM Generated ------ #########\n{response}")
            if 0 <= i < len(answers):
                logger.info(f">>> Expected Answer [similarity score w.r.t llm response = {compare_str(answers[i], response)}]: {answers[i]}")
        input("PAUSED! Press <ENTER> to continue...")
    