from dotenv import load_dotenv
load_dotenv()

import os
import yaml
import json
import argparse
import omegaconf

from openai import AzureOpenAI, OpenAI
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage

from gptcache import cache
from gptcache.adapter.langchain_models import LangChainChat
from gptcache.embedding import SBERT
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation


quote='\n\n'
top_p=1
temperature=0
max_length=200
n=1

##################### OpenAI SECRETS #####################
API_BASE = os.getenv("OPENAI_ENDPOINT", "")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_TEXT_MODEL") or os.getenv("GENCACHE_DEFAULT_MODEL", "qwen3-32b-fp8")
API_VERSION = os.getenv("API_VERSION", "")
#########################################################


def custom_last_content(data, **kwargs):
    messages = data.get("messages", [])
    if not messages:
        return ""
    last_msg = messages[-1]
    return last_msg.content


def read_multiline_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read the entire content
        content = file.read()
        
        # Split by records - each complete JSON object should be separated by a newline
        # This assumes there's a clear separator between objects
        json_strings = content.strip().split('\n{')
        
        # Process each JSON string
        for i, json_str in enumerate(json_strings):
            # Add the opening brace back except for the first item
            if i > 0:
                json_str = '{' + json_str
            
            try:
                if json_str.strip():
                    json_obj = json.loads(json_str)
                    data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON object {i+1}: {e}")
                print(f"Problematic JSON: {json_str[:100]}...")
    
    return data


def get_config(config_file):
    with open(os.path.join(os.getcwd(), config_file), 'r') as stream:
        try:
            cfg_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    cfg = omegaconf.OmegaConf.create(cfg_dict)
    return cfg


def service_enabled():
    return bool(os.getenv("GENCACHE_SERVICE_BASE_URL"))


def env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def service_client():
    return OpenAI(
        api_key=os.getenv("GENCACHE_SERVICE_API_KEY", "EMPTY"),
        base_url=os.getenv("GENCACHE_SERVICE_BASE_URL").rstrip("/"),
        timeout=float(os.getenv("GENCACHE_SERVICE_TIMEOUT", "180")),
    )


def service_chat_completion(input_dict, ground_truth=None):
    client = service_client()
    extra_body = {
        "use_cache": env_bool("GENCACHE_SERVICE_USE_CACHE", True),
        "test_mode": env_bool("GENCACHE_SERVICE_TEST_MODE", False),
        "pretrain": env_bool("GENCACHE_SERVICE_PRETRAIN", False),
        "ground_truth": ground_truth,
    }
    response = client.chat.completions.create(
        model=input_dict["model"],
        messages=input_dict["messages"],
        max_tokens=input_dict["max_tokens"],
        temperature=input_dict["temperature"],
        top_p=input_dict["top_p"],
        n=input_dict["n"],
        stop=input_dict["stop"],
        extra_body=extra_body,
    )
    content = response.choices[0].message.content or ""
    return content, getattr(response, "cached_key", None), bool(getattr(response, "cache_hit", False))


def append_result(dir_name, payload):
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        path = os.path.join(dir_name, "cache_hit_responses.jsonl")
    else:
        path = "cache_hit_responses.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--nu", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--dataset_type", type=str, default="param-w-synonym")
    parser.add_argument("--dir_name", type=str, default="")
    parser.add_argument("--feedback", type=bool, default=False)
    parser.add_argument("--scale", type=str, default="large")
    parser.add_argument("--data-file", type=str, default=None)
    args = parser.parse_args()

    # Load config
    cfg = get_config('./config.yaml')

    llm_wrapper = None
    if not service_enabled():
        from llmlib.llm_web_agent import LLMPredict

        llm_wrapper = LLMPredict(global_cache_path=cfg.global_cache.cache_path,
                                database_path=cfg.database.database_path,
                                cfg=cfg,
                                results_path=cfg.data.results_path,
                                num_records_before_caching=args.nu,
                                gamma_threshold=args.gamma,
                                use_cache=bool(cfg.global_cache.use_cache),
                                version="v2")
    
    if not os.path.exists(cfg.data.data_path):
        os.makedirs(cfg.data.data_path)

    # Load dataset
    data_file = args.data_file or f"gt_{args.dataset_type}_data_{args.scale}.jsonl"
    data = read_multiline_jsonl(data_file)
    

    if args.baseline == "gptcache":
        encoder = SBERT('all-MiniLM-L6-v2')
        # encoder = Onnx()
        data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=encoder.dimension))
        cache.init(
            pre_embedding_func=custom_last_content,
            embedding_func=encoder.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(max_distance=1.0),
            )
        cache.set_openai_key()

        llm = AzureChatOpenAI(
                    azure_endpoint=API_BASE,
                    azure_ad_token=API_KEY,
                    api_version=API_VERSION,
                    deployment_name=MODEL,
                    model_name=MODEL,
                )
        chat = LangChainChat(chat=llm)

        rationale_data = read_multiline_jsonl(f"/home/ubuntu/GenCache/LASER/gt_{args.dataset_type}_data_{args.scale}.jsonl")
        human_prompt = """You are a intelligent shopping assistant that can help users find the right item. You are given an observation of the current environment and a rationale for the next action to be taken, in the following format:

                          Current observation:
                          WebShop
                          Instruction: 
                          {user_instruction}
                          [button] Search [button_] (generate a search query based on the user instruction and select this button to find relevant items)
                          
                          Next action rationale: {rationale}
                          
                          Your task is to perform one of the function calls based on the rationale.
                          Use Search function to search for the target item in the inventory based on keywords with parameters `keyword` and `max_price` like Search(keyword="", max_price="")
                          """

    start = args.start
    end = start + args.num_examples

    final_result = {}

    for i,d in enumerate(data[start:end]):
        example_num = start + i
        if "response_action" in d:
            if args.baseline == "gptcache":
                
                rationale = rationale_data[example_num]["response_rationale"]
                prompt = human_prompt.format(user_instruction=d["instruction"], rationale=rationale)

                input_dict = {"model": MODEL,
                            "engine": MODEL,
                            "messages": [HumanMessage(content=prompt), prompt],
                            "max_tokens": max_length,
                            "temperature": temperature,
                            "top_p": top_p,
                            "n": n,
                            "stop": quote,
                            "response": d["response_action"]}
                
                print(d["instruction"])
                if service_enabled():
                    resp, cached_key, cache_hit = service_chat_completion(input_dict, d["response_action"])
                else:
                    resp, cached_key, cache_hit= llm_wrapper.llm_predict(input_dict, llm_chain=chat)

            else:
                input_dict = {"model": MODEL,
                            "engine": MODEL,
                            "messages": d["prompt_action"],
                            "max_tokens": max_length,
                            "temperature": temperature,
                            "top_p": top_p,
                            "n": n,
                            "stop": quote,
                            "response": d["response_action"]}

                print(input_dict)
                if service_enabled():
                    resp, cached_key, cache_hit = service_chat_completion(input_dict, d["response_action"])
                else:
                    resp, cached_key, cache_hit = llm_wrapper.llm_predict(input_dict)
                print("\n\n")

            final_result[example_num] = {"actual_response": d["response_action"],
                                   "llm_response": resp,
                                   "instruction": d["instruction"],
                                   "cache_hit": cache_hit}
            append_result(args.dir_name, final_result[example_num])


            # Compute and Send Feedback
            if args.feedback:

                system_prompt = """You are an expert at checking the correctness of two phrases. You will be given an instruction, and along with that two API calls which will contain phrases extracted from that instruction.  One of the API call will be the ground truth answer, and the other is an answer from our algorithm
                        The instruction will be about a human instruction to buy an item under certain price range. Both the API calls will be a Search API call where the keyword will be the item that needs to be bought and its maximum price limit.

                        Your task is to verify whether the extracted phrase from our algorithm match (corresponding to "keywords" parameter in the Search) the item description or does it not contain some important attributes that will be required if someone wants to buy that object. DO NOT look at the "max_price" parameter in the Search.
                        The ground truth response will have diferent combinations of words, which is fine, but the phrase from our algorithm should contain the information about the item that needs to be bought.

                        You will be given the information in the following format:
                        Instruction: {{instruction}}
                        Ground Truth Phrase: {{ground_truth}}
                        Algorithm Phrase: {{algorithm}}

                        Your task is to answer with "yes" if the algorithm response is correct and "no" if it is not. Do not include any other text.
                    """
                MODEL = "gpt-4.1-2025-04-14"

                verification_llm = llm = AzureOpenAI(azure_ad_token=API_KEY, azure_endpoint=API_BASE, api_version=API_VERSION)

                full_prompt = [{"role": "system", "content": system_prompt},
                              {"role": "user", "content": [{"type": "text", "text": f"Instruction: {d['instruction']}\n\nGround Truth Phrase: {d['response_action']}\n\nAlgorithm Phrase: {resp}"}]}]
            
                verification_results = {}
                try:
                    # Call the LLM
                    response = verification_llm.chat.completions.create(
                        model=MODEL,
                        messages=full_prompt,
                        max_tokens=2000,
                        temperature=0,
                        top_p=1,
                        n=1,
                        stop='\n\n' 
                    )
                    content = response.choices[0].message.content

                    if content == "no":
                        llm_wrapper.feedback(input_dict, cached_key)

                    verification_results[start] = {
                        "instruction": d["instruction"],
                        "ground_truth": d["response_action"],
                        "llm_response": resp,
                        "verification_result": content,
                        "deleted": True if content == "no" else False,
                        "dir_num": start
                    }

                    # save the jsonl result
                    with open(f"verification_results_{args.dir_name}_{MODEL}.jsonl", "a") as outfile:
                        json.dump(verification_results[start], outfile)
                        outfile.write("\n")


                except Exception as e:
                    print(e)
