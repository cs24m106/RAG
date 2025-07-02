import os
from preheader import CLONE_PATH
embed_dir = os.path.join(CLONE_PATH, "Embeddings")

import logging
logger = logging.getLogger(__name__) # Setup logging

import traceback
import numpy as np
from src.LLMs.LLM import embedding


def get_embeddings_byLLM(text_list):
    """
    Args:
        text_list: str or list of str allowed

    Returns:
        np.array of the embeddings generated for the input
    """
    try:
        vec  = embedding(text_list)
        logger.info(f"embedding [shape={vec.shape}] generated for the" + 
                    f" text: \n{repr(text_list)}" if type(text_list) == str or len(text_list) == 1 else f" text_list of len = {len(text_list)}")
        return np.array(vec, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error occurred in get_embeddings_byLLM: {e}")
        traceback.print_exc() 

def reset_series_embeddings(series_id, text_data):
    logger.info(f"reseting pre-saved Embeddings{series_id}.py with newly generated embedding")
    all_embeddings = get_embeddings_byLLM(text_data)
    np.save(os.path.join(embed_dir, f"Embeddings{series_id}.npy"), all_embeddings, allow_pickle=True)
    return all_embeddings


def get_embeddings(series_docs):
    """Add embeddings to each chunk of documents from pre-saved NumPy files."""
    logger.info(f"Getting embedding for the docs listed: {series_docs.keys()}")
    doc_key = ""
    for doc_key, doc_chunks in series_docs.items():
        # un-nest the chunks within each doc, and pull all togther as one under each series
        text_list=[]
        for chunk in doc_chunks:
            for single_chunk in chunk:
                text_list.append(single_chunk['text'])

        # try loading embedding 
        try:
            embeddings = np.load(os.path.join(embed_dir, f"Embeddings{doc_key}.npy"), allow_pickle=True)
            logger.info(f"Embeddings for {doc_key} found and loaded successfully! shape = {embeddings.shape}")
            if (len(text_list) != embeddings.shape[0]):
                logger.warning(f"no.of text-chunks [len={len(text_list)}] and no.of embeddings [shape[0]={embeddings.shape[0]}] found in savefile dont match!")
                if (embeddings.shape[0] < len(text_list)): # shallow thourgh with existing embeddings for now
                    embeddings = reset_series_embeddings(doc_key, text_list)
        except FileNotFoundError:
            logger.error(f"Embedding file for {doc_key} not found.")
            continue
        except Exception as e:
            logger.error(f"Failed to load embeddings for {doc_key}: {e}")
            continue
        
        dex ={} # map each chuck list to resp embeding w.r.t its series
        for i in range(len(text_list)):
            dex[text_list[i]] = embeddings[i]

        updated_chunks = []
        for chunk in doc_chunks:
            for idx, single_chunk in enumerate(chunk):
                try:
                    chunk[idx]['embedding'] = dex[chunk[idx]['text']]
                    updated_chunks.append(chunk[idx])
                except IndexError:
                    logger.warning(f"Embedding index {idx} out of range for {doc_key}.")
                except Exception as e:
                    logger.error(f"Error processing chunk {idx} for {doc_key}: {e}")
        
        series_docs[doc_key] = updated_chunks
    if doc_key != "":
        sample_keys = series_docs[doc_key][0].keys() if len(series_docs[doc_key])>1 else series_docs[doc_key]
        logger.info(f"Successfully added 'embeddings' to each chunk entries. each doc_key contain keys: {sample_keys}")
    else:
        logger.warning("Getting embeddings for the given series doc was unsuccessful! (given a empty series_doc list)")
    return series_docs
