import sys
import os
ROOT_DIR = "Telco-RAG_api"
curr_path = os.path.abspath(__file__)
root_path = curr_path[:curr_path.find(ROOT_DIR) + len(ROOT_DIR)]
sys.path.append(root_path)

import traceback
import git
import asyncio
import time
import argparse

from src.query import Query
from src.generate import generate, check_question
from src.LLMs.LLM import models_endpoints, models_fullnames, submit_prompt_flex
from src.LLMs.utils import update_secrets_file
from src.LLMs.settings.config import get_settings


folder_url = "https://huggingface.co/datasets/netop/Embeddings3GPP-R18"
clone_directory = os.path.join(root_path, "3GPP-Release18")

if not os.path.exists(clone_directory):
    git.Repo.clone_from(folder_url, clone_directory)
    print("Folder cloned successfully!")
else:
    print("Folder already exists. Skipping cloning.")

def TelcoRAG(query, answer= None, options= None, model_name='gpt-4o-mini', api_key= None):
    if model_name in models_fullnames:
        model_fullname = models_fullnames[model_name]
        print(f"Using model: {model_fullname}")
        endpoint = models_endpoints[model_name]
    else:
        endpoint = ""

    if api_key == None:
        try:
            settings = get_settings()
            if endpoint == "openai":
                api_key = settings.openai_api
            elif endpoint == "perplexity": 
                api_key = settings.pplx_api
            elif endpoint == "groq":
                api_key = settings.groq_api
            elif endpoint == "together":
                api_key = settings.together_api
            elif endpoint == "mistral":
                api_key = settings.mistral_api
            elif endpoint == "anthropic":
                api_key = settings.anthropic_api
            elif endpoint == "cohere":
                api_key = settings.cohere_api
            else:
                api_key = settings.any_api # defulat: for endpoint "anyscale"

            assert api_key is not None and api_key != "", "API key is missing or empty."
        except:
            sys.exit(f"You do not have an \"{endpoint}\" api key stored.")
    try:
        update_secrets_file(model_name, api_key, endpoint)
        start =  time.time()
        question = Query(query, [])

        query = question.question
        conciseprompt=f"""Rephrase the question to be clear and concise:
        
        {question.question}"""

        concisequery = submit_prompt_flex(conciseprompt, model=model_name).rstrip('"')

        question.query = concisequery

        question.def_TA_question()
        print()
        print('#'*50)
        print(concisequery)
        print('#'*50)
        print()

        question.get_3GPP_context(k=10, model_name=model_name, validate_flag=False)

        print(answer)
        if answer is not None:
            response, context , _ = check_question(question, answer, options, model_name=model_name)
            print(context)
            end=time.time()
            print(f'Generation of this response took {end-start} seconds')
            return response, question.context
        else:
            response, context, _ = generate(question, model_name)
            end=time.time()
            print(f'Generation of this response took {end-start} seconds')
            return response, context
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TelcoRAG Pipeline")
    parser.add_argument("--model", type=str, default="llama-3", help="Model name to use")
    parser.add_argument("--api", type=str, default=None, help="API key for the model")
    args = parser.parse_args()
    model_name = args.model
    api_key = args.api

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
    print("Options:")
    for key, value in question['options'].items():
        print(f"  {key}: {value}")
    print("Expected Answer:", question['answer'])
    print("Explanation:", question['explanation'])
    print("Category:", question['category'])
    print()

    # Example using an MCQ
    response, context = TelcoRAG(question['question'], question['answer'], question['options'], model_name=model_name, api_key=api_key)
    print("Generated Output given the Question & Answer and the Options along with the Query:")
    print("Response:", response)
    print("Context:", context)
    print()
    # Example using an open-end question           
    response, context = TelcoRAG(question['question'], model_name=model_name, api_key=api_key)
    print("Generated Output given only the Question and not the Options along with the Query:")
    print("Response:", response)
    print("Context:", context)