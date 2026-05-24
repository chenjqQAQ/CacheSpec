import os
import json
import random
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
import argparse

random.seed(0)

load_dotenv()


##################### OpenAI SECRETS #####################
API_BASE = os.environ["OPENAI_ENDPOINT"]
API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = os.environ["OPENAI_TEXT_MODEL"]
API_VERSION = os.environ["API_VERSION"]
#########################################################


prompt_rationale = [{'role': 'system', 'content': 'You are an intelligent shopping assistant that can help users find the right item. You are given an observation of the current web navigation session, in the following format: \n\nCurrent observation:\nWebShop\nInstruction: \n{the user instruction}\n[button] Search [button_] (generate a search query based on the user instruction and select this button to find relevant items) \n\nEvery button in the observation represents a possible action you can take. Based on the current observation, your task is to generate a rationale about the next action you should take. Note that if an history of past rationales and actions is provided, you should also consider the history when generating the rationale.\n'}, {'role': 'user', 'content': [{'type': 'text'}]}]
content_rationale = 'Current observation:\nWebShop\nInstruction: \n{}\n[button] Search [button_]'

prompt_action = [{'role': 'system', 'content': 'You are a intelligent shopping assistant that can help users find the right item. You are given an observation of the current environment and a rationale for the next action to be taken, in the following format:\n\nCurrent observation:\nWebShop\nInstruction: \n{the user instruction}\n[button] Search [button_] (generate a search query based on the user instruction and select this button to find relevant items)\n\nNext action rationale: {the rationale for the next action}\n\nYour task is to perform one of the function calls based on the rationale.\n'}, {'role': 'user', 'content': [{'type': 'text'}]}]
func_description = [{'name': 'Search', 'description': 'Use this function to search for the target item in the inventory based on keywords', 'parameters': {'type': 'object', 'properties': {'keywords': {'type': 'string', 'description': 'The keywords that describe the item to be searched for'}, 'max_price': {'type': 'string', 'description': 'The upper bound of the item price, if the upper bound is not specified, then set to 1000000.'}}, 'required': ['keywords']}}]
content_action = 'Current observation:\nWebShop\nInstruction: \n{}\n[button] Search [button_]\n\nNext action rationale:{}.'


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


def init_llm():
    llm_client = AzureOpenAI(azure_ad_token=API_KEY, azure_endpoint=API_BASE, api_version=API_VERSION)

    return llm_client




def gen_gt_response(llm, human_instr, scale):
    for i,instr in enumerate(human_instr):
        prompt_response_dir = {}
        if instr == "":
            continue
        item = {}
        print(i, instr)

        user_message = content_rationale.format(instr)
        prompt_rationale[1]['content'][0]['text'] = user_message
        response = llm_client.chat.completions.create(
            model=MODEL,
            messages=prompt_rationale,
            max_tokens=2000,
            temperature=0,
            top_p=1,
            n=1,
            stop='\n\n'
        )
        rationale = response.choices[0].message.content

        user_message = content_action.format(instr, rationale)
        prompt_action[1]['content'][0]['text'] = user_message
        response = llm_client.chat.completions.create(
            model=MODEL,
            messages=prompt_action,
            max_tokens=2000,
            temperature=0,
            top_p=1,
            n=1,
            stop='\n\n',
            functions=func_description,
            function_call=None
        )
        response2 = response.choices[0].message.dict()
        action = response2["function_call"] if response2["function_call"] is not None else None

        processed_action = f"{action['name']}({action['arguments']})"


        prompt_response_dir["instruction"] = instr
        prompt_response_dir["prompt_rationale"] = prompt_rationale
        prompt_response_dir["response_rationale"] = rationale
        prompt_response_dir["prompt_action"] = prompt_action
        prompt_response_dir["response_action"] = processed_action

        with open(f"gt_param-w-synonym_data_{scale}.jsonl", "a") as f:
            json.dump(prompt_response_dir, f, indent=4)
            f.write("\n")



if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, default="large")  # ["small", "large"]
    args = parser.parse_args()
    scale = args.scale

    llm_client = init_llm()

    if scale == "small":
        human_instr = []
        with open("human_prompts.txt", "r") as f:
            for line in f.readlines():
                human_instr.append(line.strip())
    elif scale == "large":
        instr = json.load(open("/home/ubuntu/WebShop/data/items_human_ins.json", "r"))
        human_instr = []
        for _, description in instr.items():
            instr = description[0]["instruction"][:-1]
            price = random.randint(20, 1000)
            instr = instr + f", and price lower than {price} dollars"
            human_instr.append(instr)

    gen_gt_response(llm_client, human_instr, scale)