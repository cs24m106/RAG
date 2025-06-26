import sys
import os
ROOT_DIR = "Telco-RAG_api"
curr_path = os.path.abspath(__file__)
root_path = curr_path[:curr_path.find(ROOT_DIR) + len(ROOT_DIR)]
sys.path.append(root_path)

import traceback
import git
import time
import argparse
import logging

from src.query import Query
from src.generate import generate, check_question
from src.LLMs.LLM import Mode, UPDATE_MODE, submit_prompt_flex
#from src.LLMs.utils import update_secrets_file

# Downloads
folder_url = "https://huggingface.co/datasets/netop/Embeddings3GPP-R18"
clone_directory = os.path.join(root_path, "3GPP-Release18")

if not os.path.exists(clone_directory):
    git.Repo.clone_from(folder_url, clone_directory)
    print("Folder cloned successfully!")
else:
    print("Folder already exists. Skipping cloning.")

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
        
        concisequery = submit_prompt_flex(conciseprompt, model_name, max_new_tokens).rstrip('"')

        question.query = concisequery

        question.def_TA_question()
        print()
        print('#'*50)
        print(concisequery)
        print('#'*50)
        print()

        question.get_3GPP_context(k=10, model_name=model_name, validate_flag=False)

        print(answer)
        response = None; context = None
        if answer is not None:
            response, context , _ = check_question(question, answer, options, model_name=model_name)
            print(context)
            context = question.context
        else:
            response, context, _ = generate(question, model_name)
        
        end=time.time()
        print(f'Generation of this response took {end-start} seconds')
        return response, context
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())

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
    
    print("Question:", question['question'])
    print(f"Options: [max_new_tokens={tokens}]")
    for key, value in question['options'].items():
        print(f"  {key}: {value}")
    print("Expected Answer:", question['answer'])
    print("Explanation:", question['explanation'])
    print("Category:", question['category'])
    print()

    # Example using an MCQ
    response, context = TelcoRAG(question['question'], question['answer'], question['options'], 
                                 model_name=model, mode=mode, max_new_tokens=tokens)
    print("Generated Output given the Question & Answer and the Options along with the Query:")
    print("Response:", response)
    print("Context:", context)
    print()
    # Example using an open-end question           
    response, context = TelcoRAG(question['question'], 
                                 model_name=model, mode=mode, max_new_tokens=tokens)
    print("Generated Output given only the Question and not the Options along with the Query:")
    print("Response:", response)
    print("Context:", context)