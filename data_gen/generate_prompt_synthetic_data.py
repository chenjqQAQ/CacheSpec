import os
import argparse
import random
import json
from dotenv import load_dotenv
from openai import AzureOpenAI

random.seed(0)


load_dotenv()

##################### OpenAI SECRETS #####################
API_BASE = os.environ["OPENAI_ENDPOINT"]
API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = os.environ["OPENAI_TEXT_MODEL"]
API_VERSION = os.environ["API_VERSION"]
#########################################################


prompt_structural = [{
                    'role': 'system', 
                    'content': 'You are an intelligent and capable sentence generator that can create structuralally similar sentences. You will be give a sentence which will describe a user"s request to buy a item with specific attributes and under a certain price range. Your task is to generate 10 structuralally similar sentences where the user"s request for item and param_onlys are same, but the sentence structure differs. You are free to use synonyms for few words, but some of your sentences should completely change the structure of the sentence (Example: Looking for item XYZ under ABC dollars to Not over ABC dollars, you should look for item XYZ). Please use a good mix of language, sentence rearrangements and synonyms to generate 10 such structuralally similar sentences, separated by newline character. Some of the sentences should change the structure of the original sentence, that is move the price before and the item later within the sentence.'
                    }, 
                    {'role': 'user', 
                    'content': [{'type': 'text'}]
                  }]

prompt_param_only = [{
                    'role': 'system', 
                    'content': 'You are an intelligent and capable sentence generator that can create similar sentences. You will be give a sentence which will describe a user"s request to buy a item with specific attributes and under a certain price range. Your task is to understand what item is being described, its specific attributes and the price range from the given user request. Then you should re-write the sentence in the following template\n"i want to buy {item name} with {item attributes}, under the price range of {price range}"'
                    }, 
                    {'role': 'user', 
                    'content': [{'type': 'text'}]
                   }]


def init_llm():
    llm_client = AzureOpenAI(azure_ad_token=API_KEY, azure_endpoint=API_BASE, api_version=API_VERSION)

    return llm_client


def gen_synthetic_prompts(llm, human_instr, dataset_type, scale):
    for i,instr in enumerate(human_instr):
        print(i, end='\r')
        prompt_response_dir = {}
        if instr == "":
            continue
        
        if dataset_type == "structural":
            prompt = prompt_structural
        elif dataset_type == "param_only":
            prompt = prompt_param_only
        
        prompt[1]['content'][0]['text'] = instr
        try:
            response = llm.chat.completions.create(
                model=MODEL,
                messages=prompt,
                max_tokens=2000,
                temperature=0,
                top_p=1,
                n=1,
                stop='\n\n'
            )
            content = response.choices[0].message.content.split("\n")
        except:
            continue
        prompt_response_dir["user_input"] = instr
        prompt_response_dir["similar_prompts"] = content

        with open(f"synthetic_prompts_{dataset_type}_{scale}.jsonl", "a") as f:
            json.dump(prompt_response_dir, f, indent=4)
            f.write("\n")
        


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, default="large")  # ["small", "large"]
    parser.add_argument("--data_type", type=str, default="param_only")  # ["structural", "param_only"]
    args = parser.parse_args()
    scale = args.scale
    data_type = args.data_type

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


    gen_synthetic_prompts(llm_client, human_instr, data_type, scale)

