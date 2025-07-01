import faiss
import numpy as np
import os
import traceback

import logging
logger = logging.getLogger(__name__) # Setup logging

from src.LLMs.LLM import embedding
os.environ['OMP_NUM_THREADS'] = '8'


def create_faiss_index_IndexFlatIP(embeddings, data, source):
    """
    Create FAISS IndexFlatIP from embeddings and maps indices to data and source.
    
    IndexFlatIP:
    stores all vectors in memory and allows you to search for the most similar vectors 
    to a query vector using the inner product as the similarity measure.

    faiss.IndexFlatIP(d) => creates an index for vectors of dimension d that supports fast inner product search.
    """
    logger.info("Creating IndexFlatIP...")
    try:
        d = embeddings.shape[1] # dim-0 is the no.of data entries
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        index_to_data_mapping = {i: data[i] for i in range(len(data))}
        index_to_source_mapping = {i: source[i] for i in range(len(source))}
        embedding_mapping = {i: embeddings[i] for i in range(len(embeddings))}
        logger.info(f"Creating FAISS index success! Number of vectors in index: {index.ntotal}, Dimension of vectors: {index.d}")
        return index, index_to_data_mapping, index_to_source_mapping, embedding_mapping
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        return None, None, None, None

def get_faiss_batch_index(embedded_batch):
    """Generate FAISS index from a batch of embeddings, handling missing embeddings by generating them."""
    logger.info("Generating FAISS indicies for the given embedding batch ...")
    try:
        source = [chunk['source'] for chunked_batch in embedded_batch for chunk in chunked_batch]
        embeddings = []
        data = []

        for doc in embedded_batch:
            embeddings_batch = []
            for chunk in doc:
                if 'embedding' in chunk:
                    embeddings_batch.append(chunk['embedding'])
                else:
                    embedding = generate_embedding_for_chunk(chunk)
                    if embedding is not None:
                        chunk['embedding'] = embedding
                        embeddings_batch.append(embedding)

            embeddings.extend(embeddings_batch)
            data.extend([chunk['text'] for chunk in doc])

        embeddings = np.array(embeddings, dtype=np.float32)
        return create_faiss_index_IndexFlatIP(embeddings, data, source)
    except Exception as e:
        logger.error(f"Failed to process batch for FAISS indexing: {e}")
        print(traceback.format_exc())
        return None, None, None

def generate_embedding_for_chunk(chunk):
    """Generate embeddings for a chunk using the OpenAI API."""
    try:
        response = embedding(chunk['text'])
        return response['data'][0]['embedding']
    except Exception as e:
        logger.error(f"Failed to generate embedding for chunk: {chunk['text']}. Error: {e}")
        return None
    