import toml
import os
from copy import deepcopy
from .settings.config import setting_dir, default_settings_dict

# maps endpoint to API key
which_api = { 
    "openai": "openai_api",
    "anyscale": "any_api",
    "together": "together_api",
    "groq": "groq_api",
    "mistral": "mistral_api",
    "perplexity": "pplx_api",
    "anthropic": "anthropic_api",
    "cohere": "cohere_api",
}

def update_secrets_file(model, api_key, endpoint='openai'):
    file_path = os.path.join(setting_dir, '.secrets.toml')

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            config = toml.load(file)
    else:
        config = default_settings_dict
    
    secrets =  deepcopy(config)
    secrets[which_api[endpoint]] = api_key
    config.update(secrets)
    with open(file_path, 'w') as file:
        toml.dump(config, file)
