import os
import argparse
import json


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


if __name__ == "__main__":

    parser  = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, default="large")  # ["small", "large"]
    parser.add_argument("--data_type", type=str, default="param_only")  # ["structural", "param_only"]
    args = parser.parse_args()
    scale = args.scale
    data_type = args.data_type

    normal_data = read_multiline_jsonl(f"/home/ubuntu/LASER/gt_param-w-synonym_data_{scale}.jsonl")
    synthetic_data = read_multiline_jsonl(f"/home/ubuntu/LASER/synthetic_prompts_{dataset_type}_{scale}.jsonl")

    prompt_rationale = [{'role': 'system', 'content': 'You are an intelligent shopping assistant that can help users find the right item. You are given an observation of the current web navigation session, in the following format: \n\nCurrent observation:\nWebShop\nInstruction: \n{the user instruction}\n[button] Search [button_] (generate a search query based on the user instruction and select this button to find relevant items) \n\nEvery button in the observation represents a possible action you can take. Based on the current observation, your task is to generate a rationale about the next action you should take. Note that if an history of past rationales and actions is provided, you should also consider the history when generating the rationale.\n'}, {'role': 'user', 'content': [{'type': 'text'}]}]
    content_rationale = 'Current observation:\nWebShop\nInstruction: \n{}\n[button] Search [button_]'

    prompt_action = [{'role': 'system', 'content': 'You are a intelligent shopping assistant that can help users find the right item. You are given an observation of the current environment and a rationale for the next action to be taken, in the following format:\n\nCurrent observation:\nWebShop\nInstruction: \n{the user instruction}\n[button] Search [button_] (generate a search query based on the user instruction and select this button to find relevant items)\n\nNext action rationale: {the rationale for the next action}\n\nYour task is to perform one of the function calls based on the rationale.\n'}, {'role': 'user', 'content': [{'type': 'text'}]}]
    func_description = [{'name': 'Search', 'description': 'Use this function to search for the target item in the inventory based on keywords', 'parameters': {'type': 'object', 'properties': {'keywords': {'type': 'string', 'description': 'The keywords that describe the item to be searched for'}, 'max_price': {'type': 'string', 'description': 'The upper bound of the item price, if the upper bound is not specified, then set to 1000000.'}}, 'required': ['keywords']}}]
    content_action = 'Current observation:\nWebShop\nInstruction: \n{}\n[button] Search [button_]\n\nNext action rationale:{}.'

    for i in range(len(normal_data)):
        prompt_response_dir = {}
        instr = normal_data[i]["instruction"]
        item = {}
        print(i, instr)
        user_message = content_rationale.format(instr)
        prompt_rationale[1]['content'][0]['text'] = user_message
        rationale= normal_data[i]["response_rationale"]

        user_message = content_action.format(instr, rationale)
        prompt_action[1]['content'][0]['text'] = user_message
        action = normal_data[i]["response_action"]

        prompt_response_dir["instruction"] = instr
        prompt_response_dir["prompt_rationale"] = prompt_rationale
        prompt_response_dir["response_rationale"] = rationale
        prompt_response_dir["prompt_action"] = prompt_action
        prompt_response_dir["response_action"] = action

        if dataset_type == "structural":
            with open(f"gt_{dataset_type}_data_{scale}.jsonl", "a") as f:
                json.dump(prompt_response_dir, f, indent=4)
                f.write("\n")

        similar_prompts = synthetic_data[i]["similar_prompts"]
        for p in similar_prompts:
            prompt_response_dir["instruction"] = p
            user_message = content_rationale.format(p)
            prompt_rationale[1]['content'][0]['text'] = user_message
            prompt_response_dir["prompt_rationale"] = prompt_rationale
            prompt_response_dir["response_rationale"] = rationale

            user_message = content_action.format(p, rationale)
            prompt_action[1]['content'][0]['text'] = user_message
            prompt_response_dir["prompt_action"] = prompt_action
            prompt_response_dir["response_action"] = action

            with open(f"gt_{dataset_type}_data_{scale}.jsonl", "a") as f:
                json.dump(prompt_response_dir, f, indent=4)
                f.write("\n")

