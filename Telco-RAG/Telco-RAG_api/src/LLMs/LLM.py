import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from typing import Dict, Any
import asyncio
import aiohttp
import requests
import logging
from enum import Enum

OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama local endpoint

# GLOBAL Declaration model LLM Backend MODE
class Mode(Enum):
    HuggingFace = 1
    Ollama = 2

LLM_mode = Mode.HuggingFace # private var
def UPDATE_MODE(mode:Mode): 
    global LLM_mode
    LLM_mode = mode # public meth

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    logger.debug(f"\"submit_prompt_flex_huggingface\" called with model={model_name}, max_new_tokens:{max_new_tokens}, output_json={output_json}")
    logger.debug(f"prompt:\n{prompt}")
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
    logger.debug(f"\"submit_prompt_flex_ollama\" called with model={model_name}, output_json={output_json}")
    logger.debug(f"prompt:\n{prompt}")
    
    generated = ""
    try:
        response = requests.post(OLLAMA_URL, json={"model": model_name,"prompt": prompt})
        response.raise_for_status()
        if response.status_code == 200:
            result = response.json()
            generated = result.get("response", "")
        else:
            logger.error(f"Ollama request failed: {response.status_code} {response.text}")
    except requests.RequestException as e:
        logger.error(f"FATAL: {e}")
    
    if output_json:
        generated = jsonify(generated)
    return generated

async def a_submit_prompt_flex_ollama(prompt, model_name="mistral", output_json=False):    
    logger.debug(f"\"a_submit_prompt_flex_ollama\" called with model={model_name}, output_json={output_json}")
    logger.debug(f"prompt:\n{prompt}")
    
    generated = ""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                OLLAMA_URL,
                json={"model": model_name, "prompt": prompt}
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
    logger.debug(f"\"submit_prompt_flex\" called with LLM_mode={LLM_mode}")
    if LLM_mode == Mode.HuggingFace:
        return submit_prompt_flex_huggingface(prompt, model_name, max_new_tokens, output_json)
    elif LLM_mode == Mode.Ollama:
        return submit_prompt_flex_ollama(prompt, model_name, output_json)

async def a_submit_prompt_flex(prompt: str, model_name: str = "gpt-2", max_new_tokens: int = 4096, output_json: bool = False):
    if LLM_mode == Mode.HuggingFace:
        return a_submit_prompt_flex_huggingface(prompt, model_name, max_new_tokens, output_json)
    elif LLM_mode == Mode.Ollama:
        return a_submit_prompt_flex_ollama(prompt, model_name, output_json)

def embedding(input_text, dimension=1024):
    model = SentenceTransformer("BAAI/bge-m3")  # take any embedding model here
    embeddings = model.encode([input_text], convert_to_tensor=True)
    # Ensure dimension matches (this model should have 1024 dims)
    if embeddings.shape[1] != dimension:
        raise ValueError(f"Embedding dimension mismatch: expected {dimension}, got {embeddings.shape[1]}")
    return embeddings[0].tolist()