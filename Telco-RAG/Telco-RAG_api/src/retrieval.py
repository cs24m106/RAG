import numpy as np
import traceback
from src.LLMs.LLM import embedding
from src.embeddings import get_embeddings_byLLM
import logging
logger = logging.getLogger(__name__) # Setup logging

def search_faiss_index(faiss_index, query_embedding, k=5):
    """
    Searches the FAISS Index-Database for the K-closest doc-chuck-embedding with query-embedding

    faiss_index.search: finds the k closest vectors (neighbors) in the index to the given query vector(s)
    
    returns (distances, indices):
        distances: The distances from the query to each of the k nearest neighbors.
        indices: The indices (IDs) of the nearest vectors in the FAISS index.
    """
    stop_flag = False
    if not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 1:
        logger.error("query_embedding must be a 1D numpy array")
        stop_flag = True
    if not isinstance(k, int) or k <= 0:
        logger.error("k must be a positive integer")
        stop_flag = True
    if stop_flag:
        raise ValueError()

    query_embedding_reshaped = query_embedding.reshape(1, -1) 
    # convert into 2D array format to match FAISS index

    return faiss_index.search(query_embedding_reshaped, k)

def get_query_embedding_OpenAILarge(query_text, context=None):
    if context is not None:
        if type(context) == list:
            query_text = f"{query_text}\n" + "\n".join(context)
        else:
            query_text = f"{query_text}\n {context}"
    logger.info(f"generating embeddings for the given query + {'no context' if context is None else 'context'} ...")
    return get_embeddings_byLLM(query_text)
             
def find_nearest_neighbors_faiss(query_text, faiss_index, data_mapping, k, source_mapping, embedding_mapping,  context=None):
    """
    Finds k-nearest neighbours based on FAISS search mechanism for the given query: text -> converted to -> embeddings

    returns => list of 'k' entries of each tuple containing:
        index: faiss idx of the doc-chunk from database
        data: data_map[idx] = the actual {key:'text'} data
        source: source_map[idx] = the doc-file-name of the chunk it is take from
        embedding: embedding_map[idx] = embeddings of the doc-chunk created earlier
    """
    logger.info("Finding nearest neighbours based on FAISS ...")
    try:
        query_embedding = get_query_embedding_OpenAILarge(query_text, context)
        logging.info(f"(i) generated query embeddings! shape:{query_embedding.shape}")

        D, I = search_faiss_index(faiss_index, query_embedding, k)
        logging.info(f"(ii) search_faiss_index(k={k}) success! K-Closest neighbours = Indices:{I}, Distances:{D}")

        nearest_neighbors = []
        for index in I[0]:  
            if index < len(data_mapping):  
                data = data_mapping.get(index, "Data not found")
                source = source_mapping.get(index, "Source not found")
                embedding = embedding_mapping.get(index, "Data not found")
                nearest_neighbors.append((index, data, source, embedding))
        return nearest_neighbors
    except Exception as e:
        logger.error(f"Error in find_nearest_neighbors_faiss: {str(e)}")
        traceback.print_exc()
        return []
 