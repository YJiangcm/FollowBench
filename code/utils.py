import json
import os


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
    with open(os.path.join(api_output_path, "{0}_{1}_constraint.jsonl".format(model_name, constraint_type)), 'r', encoding='utf-8') as output_file:
        for line in output_file:
            output.append(json.loads(line))

    # match
    for i in range(len(data)):
        for j in range(len(output)):
            if data[i]['instruction'] == output[j]['prompt_new']:
                data[i]['generation'] = output[j]['choices'][0]['message']['content']
                break
            if j == len(output)-1 and data[i]['level'] != 0 and data[i]['instruction'] != output[j]['prompt_new']:
                print(i)

    return data
