import argparse
import json
import os
from tqdm import tqdm
import logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from gpt4_based_evaluation import acquire_discriminative_eval_input
from openai import OpenAI

MAX_API_RETRY = 5

def get_eval(user_prompt: str, max_tokens: int, api_key: str):
    logging.basicConfig(level=logging.INFO)
    for i in range(MAX_API_RETRY):
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model='gpt-4',
                max_tokens=max_tokens,
                temperature=0.0,
                messages=[{
                    'role': 'user',
                    'content': user_prompt,
                }],
            )
            content = response['choices'][0]['message']['content']
            logger.info(content)
            return content
        except Exception as e:
            logger.error(e)
    logger.error(f'Failed after {MAX_API_RETRY} retries.')
    return 'error'


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, 'r') as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM-based evaluation.')

    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--constraint_types", nargs='+', type=str, default=['content', 'situation', 'style', 'format', 'mixed'])
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--api_output_path", type=str, default="api_output")
    parser.add_argument("--gpt4_discriminative_eval_input_path", type=str, default="gpt4_discriminative_eval_input")
    parser.add_argument("--data_gpt4_discriminative_eval_input_path", type=str, default="data_gpt4_discriminative_eval_input")
    parser.add_argument("--gpt4_discriminative_eval_output_path", type=str, default="gpt4_discriminative_eval_output")
    
    args = parser.parse_args()

    ### convert api_output to LLM_based_eval_input
    for constraint_type in args.constraint_types:
        acquire_discriminative_eval_input(
                                        data_path=args.data_path, 
                                        api_output_path=args.api_output_path, 
                                        constraint_type=constraint_type, 
                                        model_name=args.model_path, 
                                        data_gpt4_discriminative_eval_input_path=args.data_gpt4_discriminative_eval_input_path,
                                        gpt4_discriminative_eval_input_path=args.gpt4_discriminative_eval_input_path
                                        )

    ### LLM-based evaluation
    if not os.path.exists(args.gpt4_discriminative_eval_output_path):
        os.makedirs(args.gpt4_discriminative_eval_output_path)

    for constraint_type in args.constraint_types:

        eval_input = get_json_list(os.path.join(args.gpt4_discriminative_eval_input_path, "{0}_{1}_constraint.jsonl".format(args.model_path, constraint_type)))
        
        with open(os.path.join(args.gpt4_discriminative_eval_output_path, "{0}_{1}_constraint.jsonl".format(args.model_path, constraint_type)), 'w') as output_file:
            for idx in tqdm(range(len(eval_input))):
                response = get_eval(eval_input[idx]['prompt_new'], args.max_tokens, args.api_key)
                output_file.write(json.dumps({'prompt_new': eval_input[idx]['prompt_new'], "choices": [{"message": {"content": response}}]}) + '\n')
