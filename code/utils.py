import json
import os
from argparse import ArgumentParser


def convert_to_api_input(data_path, api_input_path, constraint_type):

    with open(os.path.join(data_path, "{}_constraints.json".format(constraint_type)), 'r', encoding='utf-8') as input_file:
        input_data = json.load(input_file)

    # check if the data format is correct
    num = 0
    for i in range(len(input_data)):
        if constraint_type != 'example':
            assert i % 6 == input_data[i]['level']
        if input_data[i]['level'] > 0:
            assert input_data[i]['instruction'] != ""
            num += 1
    print(f"\n[{constraint_type}] number of examples: {num}")

    with open(os.path.join(api_input_path, "{}_constraint.jsonl".format(constraint_type)), 'w', encoding='utf-8') as output_file:
        for d in input_data:
            if d['level'] > 0:
                output_file.write(json.dumps({'prompt_new': d['instruction']})+ "\n")



def data_match_api_output(data_path, api_output_path, constraint_type, model_name):
    with open(os.path.join(data_path, "{}_constraints.json".format(constraint_type)), 'r', encoding='utf-8') as data_file:
        data = json.load(data_file)

    output = []
    with open(os.path.join(api_output_path, "{0}/{1}_constraint.jsonl".format(model_name, constraint_type)), 'r', encoding='utf-8') as output_file:
        for line in output_file:
            output.append(json.loads(line))

    # match
    for i in range(len(data)):
        for j in range(len(output)):
            if data[i]['instruction'] == output[j]['prompt']:
                data[i]['generation'] = output[j]['choices'][0]['message']['content']
                break
            if j == len(output)-1 and data[i]['level'] != 0 and data[i]['instruction'] != output[j]['prompt']:
                print(i)

    return data


def add_model_args(parser: ArgumentParser):
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Hugging Face Hub model revision identifier",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "xpu"],
        default="cuda",
        help="The device type",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="A string mentioning ids of GPUs like 0,1,2 etc.",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Max Percentage of GPU memory utilised."
    )
    parser.add_argument(
        "--load-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--cpu-offloading",
        action="store_true",
        help="Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU",
    )
    parser.add_argument(
        "--gptq-ckpt",
        type=str,
        default=None,
        help="Used for GPTQ. The path to the local GPTQ checkpoint.",
    )
    parser.add_argument(
        "--gptq-wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 8, 16],
        help="Used for GPTQ. #bits to use for quantization",
    )
    parser.add_argument(
        "--gptq-groupsize",
        type=int,
        default=-1,
        help="Used for GPTQ. Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument(
        "--gptq-act-order",
        action="store_true",
        help="Used for GPTQ. Whether to apply the activation order GPTQ heuristic",
    )
    parser.add_argument(
        "--awq-ckpt",
        type=str,
        default=None,
        help="Used for AWQ. Load quantized model. The path to the local AWQ checkpoint.",
    )
    parser.add_argument(
        "--awq-wbits",
        type=int,
        default=16,
        choices=[4, 16],
        help="Used for AWQ. #bits to use for AWQ quantization",
    )
    parser.add_argument(
        "--awq-groupsize",
        type=int,
        default=-1,
        help="Used for AWQ. Groupsize to use for AWQ quantization; default uses full row.",
    )
