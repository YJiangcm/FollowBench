"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/vicuna-7b-v1.3
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
"""
import argparse
import os
import json
import torch

from fastchat.model import load_model, get_conversation_template, add_model_args

from utils import convert_to_api_input
from gpt4_based_evaluation import acquire_discriminative_eval_input


@torch.inference_mode()
def inference(args):
    # Load model
    model, tokenizer = load_model(
        args.model_path,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )

    for constraint_type in args.constraint_types:

        data = []
        with open(os.path.join(args.api_input_path, f"{constraint_type}_constraint.jsonl"), 'r', encoding='utf-8') as data_file:
            for line in data_file:
                data.append(json.loads(line))

        for i in range(len(data)):

            # Build the prompt with a conversation template
            msg = data[i]['prompt_new']
            conv = get_conversation_template(args.model_path)
            conv.append_message(conv.roles[0], msg)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Run inference
            inputs = tokenizer([prompt], return_tensors="pt").to(args.device)
            output_ids = model.generate(
                **inputs,
                do_sample=True if args.temperature > 1e-5 else False,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
            )

            if model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
            outputs = tokenizer.decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
            )

            data[i]['choices'] = [{'message': {'content': ""}}]
            data[i]['choices'][0]['message']['content'] = outputs

        # save file
        with open(os.path.join(args.api_output_path, f"{args.model_path}_{constraint_type}_constraint.jsonl"), 'w', encoding='utf-8') as output_file:
            for d in data:
                output_file.write(json.dumps(d) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--constraint_types", nargs='+', type=str, default=['content', 'scenario', 'style', 'format', 'example', 'mixed'])
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--api_input_path", type=str, default="api_input")
    parser.add_argument("--api_output_path", type=str, default="api_output")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    if not os.path.exists(args.api_input_path):
        os.makedirs(args.api_input_path)

    if not os.path.exists(args.api_output_path):
        os.makedirs(args.api_output_path)
    
    ### convert data to api_input
    for constraint_type in args.constraint_types:
        convert_to_api_input(
                            data_path=args.data_path, 
                            api_input_path=args.api_input_path, 
                            constraint_type=constraint_type
                            )

    ### model inference
    inference(args)
