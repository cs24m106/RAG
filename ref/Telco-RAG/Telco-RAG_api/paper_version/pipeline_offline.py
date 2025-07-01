import sys
sys.path.append('..') # prehead
from preheader import CLONE_PATH, RELEASE_VER, TeleQA_PATH

import os
import traceback
import git
import time
import argparse
import json
import subprocess

import random
import logging # Setup logging
logger = logging.getLogger(__name__)

from src.query import Query
from src.generate import generate, check_question
from src.LLMs.LLM import Mode, UPDATE_MODE, submit_prompt_flex
import requests

# Downloads
folder_url = F"https://huggingface.co/datasets/netop/Embeddings3GPP-R{RELEASE_VER}"

def is_valid_url(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"URL validation failed: {e}")
        return False

if not is_valid_url(folder_url):
    logger.error(f"The URL {folder_url} is not valid or not reachable. Cannot clone preprocessed-dataset!")
else:
    if not os.path.exists(CLONE_PATH):
        git.Repo.clone_from(folder_url, CLONE_PATH)
        logger.info("Folder cloned successfully!")
        try: # Download LFS files
            subprocess.run(["git", "lfs", "pull"], cwd=CLONE_PATH, check=True)
            logger.info("Git LFS files pulled successfully!")
        except Exception as e:
            logger.error(f"Failed to pull LFS files: {e}")
    else:
        logger.info("Folder already exists. Skipping cloning.")

def pause():
    input("PAUSED! Press <ENTER> to continue...")

# RAG api Class
def TelcoRAG(query, answer=None, options=None, model_name='gpt-2', mode:Mode=Mode.HuggingFace, max_new_tokens=4096):
    UPDATE_MODE(mode) # set LLM global parameter
    try:
        #update_secrets_file(model_name, api_key, endpoint)
        start =  time.time()
        question = Query(query, [])

        query = question.question
        conciseprompt=f"""Rephrase the question to be clear and concise:
        
        {question.question}"""
        
        logger.debug(f"\n{'#'*100}")
        logger.info("Task-1 ToDo: Convert given question into 'consise query'\n")
        question.query = submit_prompt_flex(conciseprompt, model_name, max_new_tokens).rstrip('"')
        logger.debug(f"\nTask-1 Over! 'concise query' : \n{repr(question.query)}")
        logger.debug(f"{'#'*100}\n")
        
        logger.debug(f"\n{'#'*100}")
        logger.info("Task-2 ToDo: Enhance the above concise query with Terms & Abbreviations\n")
        question.def_TA_question()
        logger.debug(f"\nTask-2 Over! 'enhanced_query' : \n{repr(question.enhanced_query)}")
        logger.debug(f"{'#'*100}\n")
        
        logger.debug(f"\n{'#'*100}")
        logger.info("Task-3 ToDo: Get relavant 3GPP content for the above enhanced query\n")
        question.get_3GPP_context(k=10, model_name=model_name, validate_flag=False)
        logger.debug(f"\nTask-3 Over!")
        logger.debug(f"{'#'*100}\n")
        
        logger.info(f"Check if answer is provided: \"{'None' if answer is None else answer}\"")
        logger.debug(f"\n{'#'*100}")
        response = None; context = None
        if answer is not None:
            logger.info("Task-4 (answer is provided) ToDo: check_question(question,answer,**)\n")
            check_ans, pred_ans, _ = check_question(question, answer, options, model_name=model_name)
            response = f"[Correctness:{check_ans}] Predicted Ans: {pred_ans}, Actual Ans: {answer}"
            context = question.context
        else:
            logger.info("Task-4 (answer is NOT provided) ToDo: generate(question,**)\n")
            response, context, _ = generate(question, model_name)
        logger.debug(f"\nTask-4 Over! response & context received successfully.")
        logger.debug(f"{'#'*100}\n")

        end=time.time()
        logger.info(f'Generation of this response took {end-start} seconds')
        return response, context
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.warning(traceback.format_exc())

# main()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telco-RAG Pipeline")
    parser.add_argument("-m", "--mode", type=str, choices=["HuggingFace", "Ollama"], default="Ollama",
                        help="LLM backend mode: HuggingFace or Ollama")
    parser.add_argument("-llm", "--model", type=str, default="mistral",
                        help="Model name from Hugging Face or Ollama")
    parser.add_argument("tokens", type=int, nargs="?", default=1024, help="Max new tokens for generation")
    args = parser.parse_args()
    model = args.model
    mode = Mode[args.mode]
    tokens = args.tokens

    # example question
    question =  {
        "question": "In supporting an MA PDU Session, what does Rel-17 enable in terms of 3GPP access over EPC? [3GPP Release 17]",
        "options" : { 
        "option 1": "Direct connection of 3GPP access to 5GC",
        "option 2": "Establishment of user-plane resources over EPC",
        "option 3": "Use of NG-RAN access for all user-plane traffic",
        "option 4": "Exclusive use of a non-3GPP access for user-plane traffic"
        },
        "answer": "option 2: Establishment of user-plane resources over EPC",
        "explanation": "Rel-17 enables the establishment of user-plane resources over EPC for 3GPP access in supporting an MA PDU Session, allowing for simultaneous traffic over EPC and non-3GPP access.",
        "category": "Standards overview"
    }

    # Read TeleQA.txt and load as JSON
    with open(os.path.join(TeleQA_PATH), "r", encoding="utf-8") as f:
        data = json.load(f)

    # Randomly sample a question to run the model
    num_questions = len(data)
    rand_Qno = random.randint(0, num_questions - 1)
    question = data[f"question {rand_Qno}"]
    
    # Group all "option {no}" keys into sub dict within key "options"
    options = {}
    for k, v in question.items():
        if k.startswith("option "):
            options[k] = v
    # Remove old option keys
    for k in list(question.keys()):
        if k.startswith("option "):
            question.pop(k)
    question["options"] = options
    
    # View the sampled question
    logger.debug(f"Question[{rand_Qno}]: {question['question']}")
    logger.debug(f"Options: [max_new_tokens={tokens}]")
    for key, value in question['options'].items():
        logger.debug(f"  {key}: {value}")
    logger.debug("Expected Answer: %s", question['answer'])
    logger.debug("Explanation: %s", question['explanation'])
    logger.debug("Category: %s\n", question['category'])
    pause()

    # Model Working :: Example using an MCQ
    response, context = TelcoRAG(question['question'], question['answer'], question['options'], 
                                 model_name=model, mode=mode, max_new_tokens=tokens)
    logger.info("Generated Output given the Question & Answer and the Options along with the Query:")
    logger.info(f">>> Response: {response}")
    logger.info(f">>> Context: {context}\n")
    pause()

    # Model Working :: Example using an open-end question           
    response, context = TelcoRAG(question['question'], 
                                 model_name=model, mode=mode, max_new_tokens=tokens)
    logger.info("Generated Output given only the Question and not the Options along with the Query:")
    logger.info(f">>> Response: {response}")
    logger.info(f">>> Context: {context}\n")
    pause()

'''
### Example:
Initial enchanced_query (after Task-2: adding terms and abbreviations):
'What functionalities does 3GPP Release 17 introduce for 3GPP access in an Evolved Packet Core (EPC) when establishing an Evolved Packet Data Session (ePDU)?
Terms and Definitions:
    Evolved Packet Core: Is a framework for an evolution or migration of the 3GPP system to a higher-data-rate, lower-latency, packet-optimized system that supports, multiple RATs
Abbreviations:
    3GPP: Third Generation Partnership Project'

Final enhanced_query (after Task-3: retrieval of doc-chunk-texts):
'What functionalities does 3GPP Release 17 introduce for 3GPP access in an Evolved Packet Core (EPC) when establishing an Evolved Packet Data Session (ePDU)?
Terms and Definitions:
    Evolved Packet Core: Is a framework for an evolution or migration of the 3GPP system to a higher-data-rate, lower-latency, packet-optimized system that supports, multiple RATs
Abbreviations:
    3GPP: Third Generation Partnership Project

Based on the provided texts, it is not explicit that 3GPP Release 17 introduces specific functionalities for MA PDU Sessions in terms of 3GPP access over EPC. 
However, here are some possibilities that could be indirectly related:
    1. Federated Learning Operation support (Retrieval 1)
    2. Allowing PLMN-specific access technology combinations (Retrieval 2)
    3. Identifying resources in the HSS (Retrieval 3)
    4. Enhancements for ProSe/PC5 for C2 communication (Retrieval 4)
    5. Handling Active APNs and Error Diagnostic AVPs (Retrieval 5)
    6. Request triggers related to Access Network Charging Identifier, RAN NAS Cause Support, Usage Monitoring Control Policy, etc. (Retrieval 6)
    7. File retrieval for data collection jobs (Retrieval 7)
    8. Defining semi-major, semi-minor, vertical, orientationMajor, and PointList properties (Retrieval 8)
    9. The 5G ProSe direct link setup between the source 5G ProSe end UE and the 5G ProSe UE-to-UE relay UE (Retrieval 9)
    10. Feature negotiation for N32 Handshake service with optional features applicable between the c-SEPP and p-SEPP (Retrieval 10).'
'''