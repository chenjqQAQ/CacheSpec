from dotenv import load_dotenv
load_dotenv()

import os
import yaml
import json
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI


##################### OpenAI SECRETS #####################
API_BASE = os.environ["OPENAI_ENDPOINT"]
API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = os.environ["OPENAI_TEXT_MODEL"]
API_VERSION = os.environ["API_VERSION"]
#########################################################


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

    DIR = "responses_gptcache_sbert_attribute_large_scale"

    # Load dataset
    data = read_multiline_jsonl(f"/home/ubuntu/LASER/output_dir_gptcache_sbert_attribute_large_scale/cache_hit_{DIR}.jsonl")
    verification_results = {}

    llm = AzureOpenAI(azure_ad_token=API_KEY, azure_endpoint=API_BASE, api_version=API_VERSION)

    
    for i, example in enumerate(data):
        for key in example:
            print(key)
            
            instruction = example[key]["instruction"]
            ground_truth = example[key]["actual_response"]
            algorithm = example[key]["llm_response"]
            cache_hit = example[key]["cache_hit"]

            if not cache_hit:
                # Prepare the prompt
                full_prompt = [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": [{"type": "text", "text": f"Instruction: {instruction}\n\nGround Truth Phrase: {ground_truth}\n\nAlgorithm Phrase: {algorithm}"}]}]
            
                try:
                    # Call the LLM
                    response = llm.chat.completions.create(
                        model=MODEL,
                        messages=full_prompt,
                        max_tokens=2000,
                        temperature=0,
                        top_p=1,
                        n=1,
                        stop='\n\n' 
                    )
                    content = response.choices[0].message.content

                    verification_results[key] = {
                        "instruction": instruction,
                        "ground_truth": ground_truth,
                        "llm_response": algorithm,
                        "verification_result": content,
                        "dir_num": key
                    }

                    # save the jsonl result
                    with open(f"verification_results_{DIR}_{MODEL}.jsonl", "a") as outfile:
                        json.dump(verification_results[key], outfile)
                        outfile.write("\n")

                except Exception as e:
                    print(e)
        
