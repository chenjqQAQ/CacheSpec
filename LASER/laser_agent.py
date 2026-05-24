import argparse
import ast
import random
import string
import openai
import yaml
from dotenv import load_dotenv
load_dotenv()

import adatest
import re
import os
import sys
import omegaconf
sys.path.insert(0, '/home/ubuntu/WebShop')
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

cache_dir = 'your_own_cache_dir'
import time
import functools
from joblib import Memory
import json
from openai import AzureOpenAI, OpenAI
import backoff
from copy import deepcopy
from prompt_library import *
import numpy as np


random.seed(233)
np.random.seed(233)
CacheMemory = Memory(location=cache_dir, verbose=0)

def backoff_hdlr(details):
    # Handler from https://pypi.org/project/backoff/
    print("Backing off {wait:0.1f} seconds after {tries} tries "
          "calling function {target} with args {args} and kwargs "
          "{kwargs}".format(**details))

# with open(os.path.expanduser('your_openai_key_file'), 'r') as file:
#     openai.api_key = file.read().replace('\n', '')

##################### OpenAI SECRETS #####################
API_BASE = os.getenv("OPENAI_ENDPOINT", "")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_TEXT_MODEL") or os.getenv("GENCACHE_DEFAULT_MODEL", "qwen3-32b-fp8")
API_VERSION = os.getenv("API_VERSION", "")
#########################################################


def service_enabled():
    return bool(os.getenv("GENCACHE_SERVICE_BASE_URL"))


def env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_text_function_call(text, functions):
    if not text or not functions:
        return None
    function_names = [fn.get("name") for fn in functions if isinstance(fn, dict) and fn.get("name")]
    if not function_names:
        return None
    text = text.strip()
    match = re.search(r"\b(" + "|".join(re.escape(name) for name in function_names) + r")\s*\((.*?)\)", text, re.S)
    if match:
        name = match.group(1)
        arg_text = match.group(2)
        args = {}
        call_text = f"{name}({arg_text})"
        try:
            parsed = ast.parse(call_text, mode="eval")
            call = parsed.body
            if isinstance(call, ast.Call):
                for keyword in call.keywords:
                    if keyword.arg:
                        try:
                            args[keyword.arg] = str(ast.literal_eval(keyword.value))
                        except Exception:
                            args[keyword.arg] = ast.unparse(keyword.value)
        except Exception:
            args = {}
        if not args:
            for key, value in re.findall(
                r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)(?=,\s*[A-Za-z_][A-Za-z0-9_]*\s*=|$)",
                arg_text,
                re.S,
            ):
                args[key.strip()] = value.strip().strip("'\"")
        return [{"name": name, "arguments": json.dumps(args)}]

    asin_match = re.search(r"\b(B0[A-Z0-9]{8})\b", text)
    if asin_match and "select_item" in function_names:
        return [{"name": "select_item", "arguments": json.dumps({"item_id": asin_match.group(1)})}]

    button_match = re.search(r"\[button\]\s*(.*?)\s*\[button_\]", text)
    if button_match:
        label = button_match.group(1).strip()
        label_map = {
            "Search": "Search",
            "Next >": "Next",
            "< Prev": "Prev",
            "Back to Search": "Back_to_Search",
            "Description": "Description",
            "Features": "Features",
            "Reviews": "Reviews",
            "Buy Now": "Buy_Now",
        }
        if label in label_map and label_map[label] in function_names:
            if label_map[label] == "Search":
                rest = text[button_match.end():]
                arg_match = re.search(r"\((.*?)\)", rest, flags=re.S)
                quoted = re.findall(r'"([^"]+)"', rest)
                single_quoted = re.findall(r"'([^']+)'", rest)
                keywords = ""
                if arg_match:
                    keywords = arg_match.group(1).strip().strip("'\"")
                if quoted:
                    keywords = quoted[-1]
                elif single_quoted:
                    keywords = single_quoted[-1]
                if keywords:
                    price_match = re.search(
                        r"(?:under|lower than|less than|max(?:imum)? price)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)",
                        text.lower(),
                    )
                    args = {"keywords": keywords}
                    if price_match:
                        args["max_price"] = price_match.group(1)
                    return [{"name": "Search", "arguments": json.dumps(args)}]
            return [{"name": label_map[label], "arguments": "{}"}]
        if "select_item" in function_names and re.match(r"^B0[A-Z0-9]{8}$", label):
            return [{"name": "select_item", "arguments": json.dumps({"item_id": label})}]
    return None


def append_function_format_instructions(messages, functions, function_call=None):
    if not functions:
        return messages
    schema_lines = []
    for fn in functions:
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        required = set((fn.get("parameters") or {}).get("required") or [])
        args = []
        for key in props:
            marker = "" if key in required else " optional"
            args.append(f"{key}=<value>{marker}")
        arg_text = ", ".join(args)
        schema_lines.append(f"- {fn['name']}({arg_text})")
    if not schema_lines:
        return messages
    if isinstance(function_call, dict) and function_call.get("name"):
        allowed = f"Only output this function: {function_call['name']}."
    else:
        allowed = "Choose exactly one function from the list."
    instruction = (
        "\n\nAvailable function calls:\n"
        + "\n".join(schema_lines)
        + "\n"
        + allowed
        + "\nReturn exactly one function call as plain text, for example "
        "Search(keywords='wireless mouse', max_price='30') or select_item(item_id='B012345678'). "
        "Do not output button labels, JSON, Markdown, or any extra explanation."
    )
    patched = deepcopy(messages)
    for msg in patched:
        if msg.get("role") == "system":
            msg["content"] = str(msg.get("content", "")) + instruction
            return patched
    patched.insert(0, {"role": "system", "content": instruction.strip()})
    return patched


def synthesize_function_call_from_text(text, functions, forced_call=None):
    if not text or not functions:
        return None
    parsed = parse_text_function_call(text, functions)
    if parsed is not None:
        return parsed
    function_names = [fn.get("name") for fn in functions if isinstance(fn, dict) and fn.get("name")]
    if not function_names:
        return None
    forced_name = forced_call.get("name") if isinstance(forced_call, dict) else None
    lowered = text.lower()
    name = forced_name
    if name is None:
        aliases = {
            "Search": ["search", "[button_]", "keyword"],
            "select_item": ["select_item", "select item", "item_id", "click"],
            "Next": ["next"],
            "Prev": ["prev", "previous", "back"],
            "Back_to_Search": ["back_to_search", "back to search"],
            "Buy_Now": ["buy_now", "buy now", "purchase"],
            "Description": ["description"],
            "Features": ["features"],
            "Reviews": ["reviews"],
        }
        for candidate in function_names:
            if any(alias in lowered for alias in aliases.get(candidate, [candidate.lower()])):
                name = candidate
                break
    if name is None and len(function_names) == 1:
        name = function_names[0]
    if name not in function_names:
        name = function_names[0]

    quoted = re.findall(r'"([^"]+)"', text)
    single_quoted = re.findall(r"'([^']+)'", text)
    argument_text = quoted[-1] if quoted else (single_quoted[-1] if single_quoted else "")
    args = {}
    if name == "Search":
        args["keywords"] = argument_text or text.replace("Function call:", "").strip()
        price_match = re.search(r"(?:under|lower than|less than|max(?:imum)? price)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", lowered)
        if price_match:
            args["max_price"] = price_match.group(1)
    elif name == "select_item":
        item_match = re.search(r"\b(B0[A-Z0-9]{8})\b", text)
        args["item_id"] = item_match.group(1) if item_match else argument_text
        if not args["item_id"] and "next" in lowered and "Next" in function_names:
            name = "Next"
            args = {}
    else:
        args = {}
    return [{"name": name, "arguments": json.dumps(args)}]


def function_call_has_required_args(action, functions):
    if not action or not functions:
        return True
    try:
        name = action[0].get("name")
        args = json.loads(action[0].get("arguments") or "{}")
    except Exception:
        return False
    for fn in functions:
        if isinstance(fn, dict) and fn.get("name") == name:
            required = (fn.get("parameters") or {}).get("required") or []
            return all(key in args and args[key] not in (None, "") for key in required)
    return True


class OpenAIModel(adatest.Model):
    def __init__(self, model="gpt-4-0613", quote="\"", temperature=0.7, top_p=1, max_length=30, n=1, cfg=None, logprobs=None):

        self.model_name = model
        self.model = model
        self.api_key = openai.api_key
        self.quote = quote
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.n = n
        self.logprobs = logprobs
        self.use_service = service_enabled()
        if self.use_service:
            self.service_client = OpenAI(
                api_key=os.getenv("GENCACHE_SERVICE_API_KEY", "EMPTY"),
                base_url=os.getenv("GENCACHE_SERVICE_BASE_URL").rstrip("/"),
                timeout=float(os.getenv("GENCACHE_SERVICE_TIMEOUT", "180")),
            )
            self.llm_client = None
            self.llm_wrapper = None
            return

        from llmlib.llm_web_agent import LLMPredict

        self.llm_client = AzureOpenAI(
                                    azure_endpoint=API_BASE,
                                    azure_ad_token=API_KEY,
                                    api_version=API_VERSION,
                                )
        # self.llm_client = AzureOpenAI(api_key=API_KEY, 
        #                               azure_endpoint=API_BASE, 
        #                               api_version=API_VERSION)
        self.llm_wrapper = LLMPredict(global_cache_path=cfg.global_cache.cache_path, 
                                      database_path=cfg.database.database_path, 
                                      cfg=cfg,
                                      results_path=cfg.data.results_path,
                                      num_records_before_caching=10,
                                      use_cache=bool(cfg.global_cache.use_cache),
                                      version="v1")

    @functools.lru_cache(maxsize=None)
    # @CacheMemory.cache
    @backoff.on_exception(backoff.expo,
                          (openai.RateLimitError,
                           openai.APIConnectionError),
                          max_time=1000,
                          on_backoff=backoff_hdlr)
    def __call__(self, messages, functions=None, function_call=None, salient=True):
        time.sleep(1)
        messages = eval(messages)
        if functions is not None:
            functions = eval(functions)
        if function_call is not None:
            function_call = eval(function_call)
        if not salient:
            print ('messages', messages)
        inputs = {"model": self.model,
                      "engine": self.model,
                      "messages": messages,
                      "max_tokens": self.max_length,
                      "temperature": self.temperature,
                      "top_p": self.top_p,
                      "n": self.n,
                      "stop": self.quote,
                      "functions": functions,
                      "function_call": function_call}
        
        print("Asking LLM for prediction\n------------------------")
        print(inputs["functions"], inputs["messages"])
        if self.use_service:
            kwargs = {
                "model": inputs["model"],
                "messages": inputs["messages"],
                "max_tokens": inputs["max_tokens"],
                "temperature": inputs["temperature"],
                "top_p": inputs["top_p"],
                "n": inputs["n"],
                "stop": inputs["stop"],
                "extra_body": {
                    "use_cache": env_bool("GENCACHE_SERVICE_USE_CACHE", True),
                    "test_mode": env_bool("GENCACHE_SERVICE_TEST_MODE", False),
                    "pretrain": env_bool("GENCACHE_SERVICE_PRETRAIN", False),
                },
            }
            pass_functions = env_bool("GENCACHE_WEBSHOP_PASS_FUNCTIONS", False)
            service_messages = inputs["messages"]
            if not pass_functions and inputs["functions"] is not None:
                service_messages = append_function_format_instructions(
                    service_messages,
                    inputs["functions"],
                    inputs["function_call"],
                )
                kwargs["messages"] = service_messages
            if pass_functions and inputs["functions"] is not None:
                kwargs["functions"] = inputs["functions"]
                kwargs["function_call"] = inputs["function_call"] if inputs["function_call"] is not None else "auto"
            response = self.service_client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            content = [msg.content] if msg.content is not None else [None]
            action = [msg.function_call.model_dump()] if pass_functions and getattr(msg, "function_call", None) is not None else None
            if inputs["functions"] is not None and (
                action is None or not function_call_has_required_args(action, inputs["functions"])
            ):
                synthesized = synthesize_function_call_from_text(msg.content or "", inputs["functions"], inputs["function_call"])
                if synthesized is not None:
                    action = synthesized
            print(content, action)
            return content, action

        if inputs["functions"] is not None:
            resp, cache_hit = self.llm_wrapper.llm_predict(inputs, llm_chain=self.llm_client)
            if isinstance(resp, str):
                try:
                    resp = eval(resp)
                except:
                    try:
                        resp = json.loads(resp)
                    except:
                        print("Error on Processing LLM Response")
                        exit(1)
            print("##### Response #####")
            print(resp)

            content = [resp["content"]]
            action = [resp["function_call"]] if resp["function_call"] is not None else None

        else:
            response = self.llm_client.chat.completions.create(
                        model=inputs["model"],
                        messages=inputs["messages"],
                        max_tokens=inputs["max_tokens"],
                        temperature=inputs["temperature"],
                        top_p=inputs["top_p"],
                        n=inputs["n"],
                        stop=inputs["stop"],
                    )
            resp = response.choices[0].message.dict()

            print("##### Response #####")
            print(resp)

            content = [resp["content"]]
            action = [resp["function_call"]] if resp["function_call"] is not None else None
            '''
            content = [x["message"]['content'] for x in resp['choices']]
            if 'function_call' in resp['choices'][0]['message'] and resp['choices'][0]['message']['function_call'] is not None:
                action = [x["message"]['function_call'] for x in resp['choices']]
            else:
                action = None
            '''
        
        if not salient:
            print (resp)

        print(content, action)

        return content, action

def get_config(config_file):
    with open(os.path.join(os.getcwd(), config_file), 'r') as stream:
        try:
            cfg_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    cfg = omegaconf.OmegaConf.create(cfg_dict)
    return cfg

def get_env(num_products=0):
    num_products = None if num_products in (0, None) else num_products
    env = WebAgentTextEnv(
            observation_mode='text_rich',
            render=False,
            num_products=num_products, # 1000 for small product space, None for full product space
            human_goals=True,
        )
    return env

def parse_customization(response):
    if ':' in response:
        options = response.split(':')[1].strip().split(',')
    else:
        options = response.split(',')
    options = [o.strip() for o in options]
    options = [o for o in options if o.lower() != 'none']
    return options

def auxilary_get_action(model, ob, response, available_options, setup, force=None):
    for i in range(3):
        aux_prompt = deepcopy(chat_zero_shot_mapping_action_prompt_gpt4)
        aux_prompt[0]['content'] = aux_prompt[0]['content'] % (setup[0], setup[1], setup[3])
        aux_prompt[-1]['content'][0]['text'] = aux_prompt[-1]['content'][0]['text'] % ob + '\n\n' + 'Next action rationale: ' + response[0].replace('\'', '')
        #print ('Before auxiliary prompt', response)
        if not force:
            print(aux_prompt, available_options)
            response, action = model(str(aux_prompt), functions=str(available_options))
        else:
            response, action = model(str(aux_prompt), functions=str(available_options), function_call=force)
        if action is not None:
            return response, action
        else:
            try:
                if type(eval(response[0])) == dict:
                    #print (response)
                    res = eval(response[0])
                    #print ('The action is returned in response')
                    keys = list(res.keys())
                    action = [{'name': res[keys[0]].replace('functions.', ''), 'arguments': str(res[keys[1]])}]
                    return response, action
            except:
                pass
    return None, None

def item_page_agent(env, ob, model, item_config):
    available_options = [description, reviews, features, buy_item, previous_page]
    additional_info = []
    mapping = {'Description': description, 'Reviews': reviews, 'Features': features}
    name2str = {'Description': 'description', 'Reviews': 'reviews', 'Features': 'features'}
    thinking = []
    item_text = [ob.strip().split('\n')[-7].strip(), ob.strip().split('\n')[-6].strip()]
    print (ob)
    while len(available_options) > 1:
        prompt = deepcopy(chat_zero_shot_indiv_prompt_gpt4)
        item_details = ["Target item details:"] + [f"{k}: {v}" for k, v in item_config.items()]
        item_details = '\n'.join(item_details)
        if len(additional_info) > 0:
            ob = '\n'.join([ob] + additional_info)
        ob = 'Current observation:\n' + ob + '\n' + item_details
        prompt[0]['content'] = prompt[0]['content'] % tuple(web_shop_verify_gpt4)
        prompt[-1]['content'][0]['text'] = prompt[-1]['content'][0]['text'] % ob
        response, action = model(str(prompt))
        response = post_process_response(response)
        if response[0] is not None:
            thinking.extend(response)
        if action is None:
            _, action = auxilary_get_action(model, ob, response, available_options, web_shop_verify_gpt4)
            if action == None:
                # This probably only happens when the model wanted to do customization, which we do not provide at this step
                action = [{'name': 'Buy_Now'}]
        if action[0]['name'] == 'Buy_Now':
            custom_prompt = deepcopy(chat_zero_shot_custom_prompt)
            custom_prompt[-1]['content'][0]['text'] = custom_prompt[-1]['content'][0]['text'] % ob
            tmp_response, _ = model(str(custom_prompt))
            options = parse_customization(tmp_response[0])
            arg_list = {}
            for op in options:
                arg_list[op] = {'type': 'string', 'description': f"The {op} of the item"}
            buy_item_final['parameters']['properties'] = arg_list
            buy_item_final['parameters']['required'] = list(arg_list.keys())
            prompt = deepcopy(chat_zero_shot_indiv_prompt_gpt4)
            web_shop_buy = deepcopy(web_shop_verify_gpt4)
            web_shop_buy[-1] = """At this stage, you have found the correct item. You task is to generate the correct customization options of the current item to best match the user instruction. Prepare your response in the following format:
Rationale: the user wanted {keywords of the target item}, and they required the following customization options: {cutomization of the target item}, the current item has the following customization options: {options available for the current item}, thus we should choose {the correct customization options}"""
            prompt[0]['content'] = prompt[0]['content'] % tuple(web_shop_buy)
            prompt[-1]['content'][0]['text'] = prompt[-1]['content'][0]['text'] % ob
            final_response, action = model(str(prompt))
            if action is None:
                _, action = auxilary_get_action(model, ob, final_response, str([buy_item_final]), web_shop_buy, force=str({'name': 'Buy_Now'}))
            if action is not None:
                selections = eval(action[0]['arguments'])
                for k, v in selections.items():
                    if v:
                        act = f"click[{v}]"
                        ob, rew, done, _ = env.step(act)
            act = "click[Buy Now]"            
            return act, thinking, item_text+thinking[-1:]
        elif action[0]['name'] == 'Prev':
            act = "click[< Prev]"
            print ('********* page agent decide to go back')
            return act, thinking, item_text+thinking[-1:]
        elif action[0]['name'] == 'Back_to_Search':
            act = "click[Back to Search]"
            return act, thinking, item_text+thinking[-1:]
        else:
            if action[0]['name'] not in mapping:
                continue
            act_name = mapping[action[0]['name']]
            if act_name in available_options:
                available_options.remove(act_name)
            else:
                #print ('the option is not available', action[0]['name'])
                act = "click[< Prev]"
                print ('********* page agent forced to go back')
                return act, thinking, item_text+thinking[-1:]
            act = f"click[{name2str[action[0]['name']]}]"
        ob, rew, done, _ = env.step(act)
        template = r"\[button\] < Prev \[button_\]\n(.+)"
        match = re.search(template, ob)
        if match is not None:
            new_info = re.search(template, ob).group(1).strip()
        else:
            new_info = 'None'
        additional_info.append(f"{name2str[action[0]['name']]}:\n{new_info}\n")
        ob, rew, done, _ = env.step("click[< Prev]")

def post_process_response(response):
    response = response[0]
    response = response.replace('Rationale:', '').strip()
    response = response.replace('Feedback:', '').strip()
    pattern = r"Rationale\d: "
    response = re.sub(pattern, '', response).strip()
    return [response]

def print_prompt(prompt):
    for turn in prompt:
        print (turn['content'])

def get_action(action, mapping):
    action[0]['arguments'] = action[0]['arguments'].replace('null', 'None').replace('false', 'False').replace('true', 'True')
    act_name = mapping[action[0]['name']]
    act_arg = eval(action[0]['arguments'])
    return act_name, act_arg

def back_up_agent(env, session, model, browsed_items, instruction):
    fake_ob = f'Current observation:\nInstruction:\n{instruction}\n'
    for k, v in browsed_items.items():
        fake_ob += f'[button] {k} [button_]\n'
        fake_ob += f'{v[0]}\n{v[1].replace("Price: ", "")}\n'
    print ('-----------------Back up agent-----------------')
    prompt = deepcopy(chat_zero_shot_indiv_prompt_gpt4)
    web_shop_backup = deepcopy(web_shop_select_gpt4)
    fake_layout = web_shop_backup[3].split('\n')
    fake_layout = '\n'.join(fake_layout[:2] + fake_layout[5:])
    web_shop_backup[3] = fake_layout
    web_shop_backup[4] = 'At this stage, you should identify one of the items on the current page that best matches the user instruction. If none of the items match the user instruction, identify the item that is the closest match to the user instruction.'
    prompt[0]['content'] = prompt[0]['content'] % tuple(web_shop_backup)
    prompt[-1]['content'][0]['text'] = prompt[-1]['content'][0]['text'] % fake_ob 
    response, action = model(str(prompt))
    response = post_process_response(response)
    if action is None:
        response, action = auxilary_get_action(model, fake_ob, response, str([click_item]), web_shop_select_gpt4, force=str({'name': 'select_item'}))
    best_item = ''
    if action is not None:
        try:
            action_args = eval(action[0].get('arguments', '{}'))
            if isinstance(action_args, dict):
                best_item = action_args.get('item_id', '')
        except Exception:
            best_item = ''
    if best_item not in browsed_items:
        if not browsed_items:
            return 0
        best_item = list(browsed_items.keys())[0]
        print ('Back up agent failed to select any item')
    (ob, _) = env.reset(session=session)
    print('best item', best_item)
    print('browsed items', browsed_items)
    print(browsed_items[best_item])
    ob, rew, done, _ = env.step(f'search[{browsed_items[best_item][0]}]')
    ob, rew, done, _ = env.step(f'click[{best_item}]')

    custom_prompt = deepcopy(chat_zero_shot_custom_prompt)
    custom_prompt[-1]['content'][0]['text'] = custom_prompt[-1]['content'][0]['text'] % ob
    tmp_response, _ = model(str(custom_prompt))
    options = parse_customization(tmp_response[0])
    arg_list = {}
    for op in options:
        arg_list[op] = {'type': 'string', 'description': f"The {op} of the item"}
    buy_item_final['parameters']['properties'] = arg_list
    buy_item_final['parameters']['required'] = list(arg_list.keys())
    prompt = deepcopy(chat_zero_shot_indiv_prompt_gpt4)
    web_shop_buy = deepcopy(web_shop_verify_gpt4)
    web_shop_buy[-1] = """At this stage, you have found the correct item. You task is to generate the correct customization options of the current item to best match the user instruction. Prepare your response in the following format:
Rationale: the user wanted {keywords of the target item}, and they required the following customization options: {cutomization of the target item}, the current item has the following customization options: {options available for the current item}, thus we should choose {the correct customization options}"""
    prompt[0]['content'] = prompt[0]['content'] % tuple(web_shop_buy)
    prompt[-1]['content'][0]['text'] = prompt[-1]['content'][0]['text'] % ob
    response, action = model(str(prompt))
    response = post_process_response(response)
    if action is None:
        response, action = auxilary_get_action(model, ob, response, str([buy_item_final]), web_shop_buy, force=str({'name': 'Buy_Now'}))
    action[0]['arguments'] = action[0]['arguments'].replace('null', 'None').replace('false', 'False').replace('true', 'True')
    selections = eval(action[0]['arguments'])
    for k, v in selections.items():
        if v:
            act = f"click[{v}]"
            ob, rew, done, _ = env.step(act)
    act = "click[Buy Now]"
    ob, rew, done, _ = env.step(act)
    return rew  

def indiv_prompt_agent(agent_args, cfg):
    env = get_env(agent_args.num_products)
    max_iter = 13
    rewards = []
    term_status = []
    model = OpenAIModel(model=agent_args.model_name, quote='\n\n', temperature=0, max_length=200, n=1, cfg=cfg)
    # s = [{'role': 'system', 'content': 'You are an intelligent shopping assistant that can help users find the right item. You are given an observation of the current web navigation session, in the following format: \n\nCurrent observation:\nWebShop\nInstruction: \n{the user instruction}\n[button] Search [button_] (generate a search query based on the user instruction and select this button to find relevant items) \n\nEvery button in the observation represents a possible action you can take. Based on the current observation, your task is to generate a rationale about the next action you should take. Along with the rationale, you should also clearly include the action that should be taken. Note that if an history of past rationales and actions is provided, you should also consider the history when generating the rationale.\n'}, {'role': 'user', 'content': [{'type': 'text', 'text':'Current observation:\nWebShop\nInstruction: \ni am looking for blue color toothbrushes that helps to maintain my oral hygiene, and price lower than 50.00 dollars\n[button] Search [button_]'}]}]
    # print(model(str(s)))
    start = agent_args.start
    end = start + agent_args.num_examples
    episodes_len = {}
    
    for session in range(start, end):
        (ob, _) = env.reset(session=session)
        available_funcs = None
        forced_funcs = None
        mapping = {'Search': 'search', 'select_item': 'click', 'Next': 'next', 'Prev': 'prev', 'Back_to_Search': 'back',
                'search_item_with_history': 'search'}
        history = ['History:']
        thinking = []
        item_config = None
        browsed_items = {}
        bought = False
        instruction = None 
        counter = 0
        while counter < max_iter:
            print("")
            ob = '\n'.join(ob.strip().split('\n\n'))
            ob = 'Current observation:\n' + ob
            # print(ob)
            prompt = deepcopy(chat_zero_shot_indiv_prompt_gpt4) 
            if 'Back to Search' in ob:
                # if counter == 1:
                #     print(ob)
                available_funcs = [click_item, next_page, back_to_search]
                setup = web_shop_select_gpt4
            else:
                if not instruction:
                    instruction = ob.split('\n')[3].strip()
                    print (instruction)
                if len(history) == 1:
                    available_funcs = [search_items]
                    setup = web_shop_search_gpt4
                else:
                    available_funcs = [search_items]
                    setup = web_shop_search_gpt4
                    ob = ob + '\n' + '\n'.join(history)
            
            prompt[0]['content'] = prompt[0]['content'] % tuple(setup)
            prompt[-1]['content'][0]['text'] = prompt[-1]['content'][0]['text'] % ob 
            ############ PATCH ##############
            prompt[-1]['content'][0]['text'] = prompt[-1]['content'][0]['text'].replace('\'', '')
            #################################
            print("Prompt:")
            print(prompt)

            if 'Back to Search' not in ob and len(history) > 1:
                print_prompt(prompt)
                history = ['History:']
            print(prompt)
            response, action = model(str(prompt))
            # print("------- Model Response -------")
            # print(f"Response: {response}")
            response = post_process_response(response)
            if response[0] is not None:
                thinking.extend(response)
            else:
                thinking.extend(['None'])
            if action is None:\
                _, action = auxilary_get_action(model, ob, response, available_funcs, setup)
            act_name, act_arg = get_action(action, mapping)
            print("Action: ", act_name, act_arg)
            if act_name == 'search':
                args = ', '.join([f"{k}='{v}'" for k, v in act_arg.items()])
                history.append(f'Rationale{int((len(history)-1)/2)}: {thinking[-1]}')
                history.append(f'Action{int((len(history)-1)/2)}: {action[0]["name"]}({args})')
                item_config = act_arg
                act = f"{act_name}[{act_arg['keywords']}]"
            elif act_name == 'click' and act_arg['item_id'] != 'next':
                print ('going to check', act_arg['item_id'])
                counter += 1
                if act_arg['item_id'] in browsed_items:
                    print ('Agent attempt to click an item that has been clicked before')
                    aux_prompt = deepcopy(chat_zero_shot_mapping_action_prompt_gpt4)
                    aux_prompt[0]['content'] = aux_prompt[0]['content'] % (setup[0], setup[1], setup[3])
                    aux_prompt[-1]['content'][0]['text'] = aux_prompt[-1]['content'][0]['text'] % ob + '\n\n' + 'Next action rationale: ' + f'items that have been clicked before do not match the user instruction, so the next action should be select a different item that have not been clicked before.'
                    response, action = model(str(aux_prompt), functions=str(available_funcs), function_call=forced_funcs)
                    act_name = mapping[action[0]['name']]
                    act_arg = eval(action[0]['arguments'])
                if act_name == 'next':
                    act = "click[Next >]"
                    history.append(f'Rationale{int((len(history)-1)/2)}: {thinking[-1]}')
                    history.append(f'Action{int((len(history)-1)/2)}: next_page()')
                else:
                    history.append(f'Rationale{int((len(history)-1)/2)}: {thinking[-1]}')
                    history.append(f'Action{int((len(history)-1)/2)}: {action[0]["name"]}({act_arg["item_id"]})')
                    act = f"{act_name}[{act_arg['item_id']}]"
                    ob, rew, done, _ = env.step(act)
                    act, rationale, item_text = item_page_agent(env, ob, model, item_config)
                    thinking.extend(rationale)
                    browsed_items[act_arg['item_id']] = item_text
                    if act == "click[< Prev]":
                        history.append(f'Rationale{int((len(history)-1)/2)}: {thinking[-1]}')
                        history.append(f'Action{int((len(history)-1)/2)}: previous_page()')
            elif act_name == 'next':
                act = "click[Next >]"
                history.append(f'Rationale{int((len(history)-1)/2)}: {thinking[-1]}')
                history.append(f'Action{int((len(history)-1)/2)}: next_page()')
            elif act_name == 'prev':
                act = "click[< Prev]"
                history.append(f'Rationale{int((len(history)-1)/2)}: {thinking[-1]}')
                history.append(f'Action{int((len(history)-1)/2)}: previous_page()')
            elif act_name == 'back':
                act = "click[Back to Search]"
                history.append(f'Rationale{int((len(history)-1)/2)}: {thinking[-1]}')
                history.append(f'Action{int((len(history)-1)/2)}: back_to_search()')
            else:
                print ('encountered unknown action', act_name)
                exit()
            counter += 1
            ob, rew, done, _ = env.step(act)
            if act == 'click[Buy Now]':
                bought = True
                break
            #print ('iter', counter)
        if not bought:
            rew = back_up_agent(env, session, model, browsed_items, instruction)
            episodes_len[session] = -1 
        else:
            episodes_len[session] = counter
        rewards.append(rew)
        term_status.append(rew==1)
        print (f'episode {session}, reward {rew}, term_status {rew==1}')
    print('reward', np.mean(rewards), 'term_status', np.mean(term_status))
    results_dir = os.getcwd()
    if getattr(agent_args, "output_dir", None):
        results_dir = os.path.join(agent_args.output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f'{agent_args.model_name}_rewards_{start}-{end}_max13.json'), 'w') as fout:
        json.dump(rewards, fout)
    with open(os.path.join(results_dir, f'{agent_args.model_name}_episodes_len_{start}-{end}_max13.json'), 'w') as fout:
        json.dump(episodes_len, fout)
    summary = {
        "model_name": agent_args.model_name,
        "start": start,
        "end": end,
        "num_examples": agent_args.num_examples,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "success_rate": float(np.mean(term_status)) if term_status else 0.0,
        "num_products": agent_args.num_products,
    }
    with open(os.path.join(results_dir, "webshop_summary.json"), "w") as fout:
        json.dump(summary, fout, indent=2)
    return rewards

if __name__ == '__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="webshop")
    parser.add_argument("--model_name", type=str, default=MODEL)
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_products", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()
    cfg = get_config('./config.yaml')
    indiv_prompt_agent(args, cfg)
