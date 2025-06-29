import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from typing import Dict, Any
import asyncio
import aiohttp
import requests
import logging
import json
from enum import Enum

OLLAMA_URL = "http://localhost:11434/api/"  # Ollama local endpoint

# GLOBAL Declaration model LLM Backend MODE
class Mode(Enum):
    HuggingFace = 1
    Ollama = 2

LLM_mode = Mode.HuggingFace # private var
def UPDATE_MODE(mode:Mode): 
    global LLM_mode
    LLM_mode = mode # public meth

logger = logging.getLogger(__name__) # Setup logging

# Mapping of model names to Hugging Face repo IDs
models_ids = {
    "gpt-2": "openai-community/gpt2",
    "gpt-3": "facebook/opt-125m",
    "deepseek": "deepseek-ai/DeepSeek-V3",
    "mistral-small": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "mistral-large": "mistralai/Mistral-Large-Instruct-2411",
    "code-llama": "codellama/CodeLlama-70b-Instruct-hf",
    "phi": "microsoft/phi-4",
    "command-R+": "CohereLabs/c4ai-command-r-plus-08-2024",
    "pplx": "perplexity-ai/r1-1776",
    "llama-2": "meta-llama/Llama-2-70b-chat-hf",
    "llama-3": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen": "Qwen/Qwen3-235B-A22B",
    "gemma": "google/gemma-7b-it",
    "wizard": "dreamgen/WizardLM-2-8x22B",
}

# Preload all models/tokenizers (optional for performance)
model_cache: Dict[str, Any] = {}

def load_model(model_name: str):
    """Load tokenizer and model for a given Hugging Face model."""
    model_id = models_ids[model_name]
    if model_name in model_cache:
        return model_cache[model_name]

    print(f"Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Fix for missing pad token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  # Automatically use GPU if available
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    model_cache[model_name] = (tokenizer, model)
    return tokenizer, model


def submit_prompt_flex_huggingface(prompt: str, model_name: str = "gpt-2", max_new_tokens: int = 4096, output_json: bool = False):
    """
    Generate text using a Hugging Face model.
    Args:
        prompt: Input text.
        model_name: Model name from `models_ids`.
        max_new_tokens: Max tokens to generate.
        output_json: Whether to format output as JSON (basic placeholder).
    """
    logger.info(f"=> called with model={model_name}, max_new_tokens:{max_new_tokens}, output_json={output_json}, prompt: \n{repr(prompt)}")
    tokenizer, model = load_model(model_name)
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)    
    if output_json:
        response = jsonify(response)    
    return response.strip()


async def a_submit_prompt_flex_huggingface(prompt: str, model_name: str = "gpt-2", max_new_tokens: int = 4096, output_json: bool = False):
    """Async wrapper for generate_text."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, submit_prompt_flex_huggingface, prompt, model_name, max_new_tokens, output_json)
    return result


def submit_prompt_flex_ollama(prompt, model_name="mistral", output_json=False):
    """
    prompt using -> '/api/generate'
    
    NOTE: This endpoint can return streamed responses (one JSON object per line), 
    especially when used for chat or large language model generation. However, your code is using response.json(), 
    which expects a single valid JSON object — hence the Extra data error when it encounters additional JSON objects or garbage data.
    """
    logger.info(f"=> called with model={model_name}, output_json={output_json}, prompt: \n{repr(prompt)}")
    url = OLLAMA_URL + "generate" # lets use generate api for sync fn

    generated = ""
    try:
        response = requests.post(url, json={"model": model_name,"prompt": prompt})
        response.raise_for_status()
        # if response.status_code == 200: means llm had responded properly
        # Parse each line as a separate JSON object
        for line in response.iter_lines():
            if line:
                try:
                    result = json.loads(line)
                    generated += result.get("response", "")
                except json.JSONDecodeError as je:
                    logger.warning(f"Skipping non-JSON line: {line} | Error: {je}")
        generated = generated.strip()
        logger.info("ollama request successfully parsed!")
    except Exception as e:
        logger.error(f"FATAL [status:{response.status_code}]: {e}, response: {response.text}")
    
    if output_json:
        generated = jsonify(generated)
    return generated


async def a_submit_prompt_flex_ollama(prompt, model_name="mistral", output_json=False):
    """
    prompt using -> '/api/chat'
    
    NOTE: for structured chat-based responses, which *may* return a single JSON object instead of a stream.
    keep it check later for now...
    """
    logger.info(f"=> called with model={model_name}, output_json={output_json}, prompt: \n{repr(prompt)}")
    url = OLLAMA_URL + "chat" # lets use chat api for async fn

    generated = ""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={"model": model_name, "messages": [{"role": "user", "content": prompt}]}
            ) as response:
                response.raise_for_status()
                
                result = await response.json()
                generated = result.get("response", "")
                
    except aiohttp.ClientResponseError as e:
        logger.error(f"Ollama request failed: {e.status}: {e.message}")
    except Exception as e:
        logger.error(f"FATAL: {e}")
    
    if output_json:
        generated = jsonify(generated)
    return generated


def jsonify(response):
    response = response.replace('"\n', '",\n')
    response = response[:response.rfind("}")+1]
    return response


def submit_prompt_flex(prompt: str, model_name: str = "gpt-2", max_new_tokens: int = 4096, output_json: bool = False):
    logger.info(f"=> called with LLM_mode={LLM_mode}")
    if LLM_mode == Mode.HuggingFace:
        return submit_prompt_flex_huggingface(prompt, model_name, max_new_tokens, output_json)
    elif LLM_mode == Mode.Ollama:
        return submit_prompt_flex_ollama(prompt, model_name, output_json)

async def a_submit_prompt_flex(prompt: str, model_name: str = "gpt-2", max_new_tokens: int = 4096, output_json: bool = False):
    logger.info(f"=> called with LLM_mode={LLM_mode}")
    if LLM_mode == Mode.HuggingFace:
        return a_submit_prompt_flex_huggingface(prompt, model_name, max_new_tokens, output_json)
    elif LLM_mode == Mode.Ollama:
        return a_submit_prompt_flex_ollama(prompt, model_name, output_json)


def embedding(text_list:list, model_name:str ="BAAI/bge-m3", dimension:int =1024):
    logger.info(f"Finding Emdeddings based on params: [model:{model_name}, dim:{dimension}]")
    model = SentenceTransformer(model_name)  # take any embedding model here
    embeddings = model.encode(text_list, convert_to_tensor=True)
    # Ensure dimension matches (this model should have 1024 dims)
    if embeddings.shape[1] != dimension:
        raise ValueError(f"Embedding dimension mismatch: expected {dimension}, got {embeddings.shape[1]}")
    return embeddings