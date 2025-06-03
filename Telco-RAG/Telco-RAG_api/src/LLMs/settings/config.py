import pathlib
from os import listdir
from os.path import abspath, dirname, join
import toml
from dynaconf import Dynaconf

current_dir = dirname(abspath(__file__))
# setting_dir = join(current_dir, "settings")
setting_dir = current_dir

toml_files = list(pathlib.Path(join(setting_dir)).glob('*.toml'))


default_settings_dict = {
"openai_api_key" : "",
"any_api_key" : "",
"mistral_api" : "",
"anthropic_api" : "",
"cohere_api" : "",
"google_search_api" : "",
"pplx_api" : "",
"together_api" : "",
"rate_limit" : 9,
"fireworks_api" : ""
}

if not toml_files:
    default_toml_path = join(setting_dir, '.secrets.toml')
    with open(default_toml_path, 'w') as file:
        toml.dump(default_settings_dict, file)
    toml_files.append(default_toml_path)
    
global_settings = Dynaconf(
    envvar_prefix=False,
    merge_enabled=True,
    settings_files=toml_files,
)

def get_settings():
    return global_settings
