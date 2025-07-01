import openai
from src.LLMs.settings.config import get_settings
# completely removed redundant codes by re-using funs from src.LLMs.LLM.py code
from src.LLMs.LLM import submit_prompt_flex, a_submit_prompt_flex


def update_api_key():
    openai.api_key = get_settings().openai_api_key
    return

def submit_prompt_flex_UI(prompt, model="gpt-4o-mini", output_json=False):
    if openai.api_key == "":
        update_api_key()
    return submit_prompt_flex(prompt, model_name=model, output_json=output_json)


async def a_submit_prompt_flex_UI(prompt, model="gpt-4o-mini", output_json=False):
    if openai.api_key == "":
        update_api_key()
    return a_submit_prompt_flex(prompt, model_name=model, output_json=output_json)
