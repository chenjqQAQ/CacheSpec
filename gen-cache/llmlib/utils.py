from omegaconf import DictConfig
import difflib
import os
import ast
import torch
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import AzureChatOpenAI
from langchain.schema.language_model import BaseLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline


from huggingface_hub import login

os.environ["HF_HOME"] = "/datadrive/cache"
login(token="<token>")


API_BASE = os.environ["OPENAI_ENDPOINT"]
API_KEY = os.environ["OPENAI_API_KEY"]
MODEL_POWERFUL = os.environ["OPENAI_TEXT_MODEL_POWERFUL"]
MODEL = os.environ["OPENAI_TEXT_MODEL_DEFAULT"]
API_VERSION = os.environ["API_VERSION"]



def init_llm(cfg: DictConfig, enable_cache: bool, powerful_model: bool = False) -> BaseLanguageModel:
        """init the language model."""
        if cfg.global_cache.model == "gpt-4x":
            if powerful_model:
                llm = AzureChatOpenAI(
                    azure_endpoint=API_BASE,
                    azure_ad_token=API_KEY,
                    api_version=API_VERSION,
                    deployment_name=MODEL_POWERFUL,
                    model_name=MODEL_POWERFUL,
                    verbose=True,
                    temperature=cfg.llm.temperature,
                    cache=enable_cache,
                )
            else:
                llm = AzureChatOpenAI(
                    azure_endpoint=API_BASE,
                    azure_ad_token=API_KEY,
                    api_version=API_VERSION,
                    deployment_name=MODEL,
                    model_name=MODEL,
                    verbose=True,
                    temperature=cfg.llm.temperature,
                    cache=enable_cache,
                )
        elif cfg.global_cache.model == "llama":
            model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

            pipe = pipeline("text-generation", model=model, model_kwargs={"torch_dtype": torch.bfloat16}, tokenizer=tokenizer, max_new_tokens=128000)
            llm = HuggingFacePipeline(pipeline=pipe)

        return llm


def dict_to_tuple_key(dictionary):
    """ Given a dictionary, converts the dictionary to a tuple """
    hashable_dictionary = dict()
    for key, value in dictionary.items():
        if isinstance(value, list):
            continue
        hashable_dictionary[key] = value

    return tuple(sorted(hashable_dictionary.items()))


def list_of_dict_to_tuple_key(list_of_dict):
    """ Given a list of dictionaries, converts the list of dictionaries to a list of tuples """
    return tuple([dict_to_tuple_key(dictionary) for dictionary in list_of_dict])


def tuple_to_dict_key(tup):
    """ Given a tuple, converts the tuple to a dictionary """
    return dict(tup)


def replace_braces(text: str) -> str:
    # Replace already present '{{' and '}}' with placeholders
    text = text.replace('{{', '<<').replace('}}', '>>')
    
    # Replace single '{' and '}' with '{{' and '}}'
    text = text.replace('{', '{{').replace('}', '}}')
    
    # Revert placeholders back to '{{' and '}}'
    text = text.replace('<<', '{{').replace('>>', '}}')
    
    return text


def sentence_difference(prompt1, prompt2):
    diff = []
    for line in difflib.unified_diff(prompt1.split("\n"), prompt2.split("\n"), fromfile='prompt1', tofile='prompt2', lineterm=''):
        diff.append(line)
        
    return diff


def compute_token_length(str):
    return len(str.split(" "))


def is_valid_dictionary(content: str) -> bool:
    try:
        # Try parsing the string to a Python object
        parsed = ast.literal_eval(content)
        # Check if it's a dictionary and contains 'content': None or similar keys
        return isinstance(parsed, dict)
    except (ValueError, SyntaxError):
        # If it fails to parse, check if it's just an error message containing "None:"
        return False