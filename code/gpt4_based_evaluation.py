import json
import csv
import os
import re
import random
import matplotlib.pyplot as plt
import logging  
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from utils import data_match_api_output
from rule_based_evaluation import rule_evaluation, csl_five_constraint, check_match



############################################################################################################################################## evaluation prompt for content constraints

def content_evaluation_prompt(evolve_instructions, answer):
    level = len(evolve_instructions)-1

    if level == 1:
        prompt = "Given an initial instruction, we add one content constraint and obtain the final instruction with 1 additional constraint.\n\n" \
                + "#Initial Instruction#\n" + evolve_instructions[0] + "\n\n"

        prompt += "#Initial Instruction + 1 constraint#\n" + evolve_instructions[1] + "\n\n"

        prompt += "#Answer of Initial Instruction + 1 constraint#\n" + answer + "\n\n"

        prompt += "#System#\n1) Please identify the 1 added constraint.\n2) Please descriminate if the #Answer of Initial Instruction + 1 constraint# satisfies the 1 added constraint.\n3) In the final line, only output a Python LIST with 1 element ('YES' or 'NO') indicating whether the answer satisfies the 1 added constraint."


    else:
        prompt = "Given an initial instruction, we add one content constraint per time and obtain the final instruction with {} additional constraints.\n\n".format(level) \
                + "#Initial Instruction#\n" + evolve_instructions[0] + "\n\n"

        prompt += "#Initial Instruction + 1 constraint#\n" + evolve_instructions[1] + "\n\n"

        for i in range(2, level+1):
            prompt += "#Initial Instruction + {} constraints#\n".format(i) + evolve_instructions[i] + "\n\n"

        prompt += "#Answer of Initial Instruction + {} constraints#\n".format(level) + answer + "\n\n"

        prompt += "#System#\n1) Please identify all {level} added constraints.\n2) For the {level} added constraints, discriminate if the #Answer of Initial Instruction + {level} constraints# satisfies each constraint.\n3) In the final line, only output a Python LIST with {level} elements ('YES' or 'NO') indicating whether the answer satisfies each constraint.".format(level=level)

    return prompt



############################################################################################################################################## evaluation prompt for situation constraints

def situation_evaluation_prompt(evolve_instructions, answer):
    level = len(evolve_instructions)-1

    if level == 1:
        prompt = "Given an initial instruction, we add one situation constraint (information to describe a specific situation/background) and obtain the final instruction with 1 additional constraint.\n\n" \
                + "#Initial Instruction#\n" + evolve_instructions[0] + "\n\n"

        prompt += "#Initial Instruction + 1 constraint#\n" + evolve_instructions[1] + "\n\n"

        prompt += "#Answer of Initial Instruction + 1 constraint#\n" + answer + "\n\n"

        prompt += "#System#\n1) Please identify the 1 added constraint.\n2) Please descriminate if the #Answer of Initial Instruction + 1 constraint# satisfies the 1 added constraint.\n3) In the final line, only output a Python LIST with 1 element ('YES' or 'NO') indicating whether the answer satisfies the 1 added constraint."


    else:
        prompt = "Given an initial instruction, we add one situation constraint (information to describe a specific situation/background) per time and obtain the final instruction with {} additional constraints.\n\n".format(level) \
                + "#Initial Instruction#\n" + evolve_instructions[0] + "\n\n"

        prompt += "#Initial Instruction + 1 constraint#\n" + evolve_instructions[1] + "\n\n"

        for i in range(2, level+1):
            prompt += "#Initial Instruction + {} constraints#\n".format(i) + evolve_instructions[i] + "\n\n"

        prompt += "#Answer of Initial Instruction + {} constraints#\n".format(level) + answer + "\n\n"

        prompt += "#System#\n1) Please identify all {level} added constraints.\n2) For the {level} added constraints, discriminate if the #Answer of Initial Instruction + {level} constraints# satisfies each constraint.\n3) In the final line, only output a Python LIST with {level} elements ('YES' or 'NO') indicating whether the answer satisfies each constraint.".format(level=level)

    return prompt



############################################################################################################################################## evaluation prompt for style constraints

def style_evaluation_prompt(evolve_instructions, answer):
    level = len(evolve_instructions)-1

    if level == 1:
        prompt = "Given an initial instruction, we add one style constraint and obtain the final instruction with 1 additional constraint.\n\n" \
                + "#Initial Instruction#\n" + evolve_instructions[0] + "\n\n"

        prompt += "#Initial Instruction + 1 constraint#\n" + evolve_instructions[1] + "\n\n"

        prompt += "#Answer of Initial Instruction + 1 constraint#\n" + answer + "\n\n"

        prompt += "#System#\n1) Please identify the 1 added constraint.\n2) Please descriminate if the #Answer of Initial Instruction + 1 constraint# satisfies the 1 added constraint.\n3) In the final line, only output a Python LIST with 1 element ('YES' or 'NO') indicating whether the answer satisfies the 1 added constraint."


    else:
        prompt = "Given an initial instruction, we add one style constraint per time and obtain the final instruction with {} additional constraints.\n\n".format(level) \
                + "#Initial Instruction#\n" + evolve_instructions[0] + "\n\n"

        prompt += "#Initial Instruction + 1 constraint#\n" + evolve_instructions[1] + "\n\n"

        for i in range(2, level+1):
            prompt += "#Initial Instruction + {} constraints#\n".format(i) + evolve_instructions[i] + "\n\n"

        prompt += "#Answer of Initial Instruction + {} constraints#\n".format(level) + answer + "\n\n"

        prompt += "#System#\n1) Please identify all {level} added constraints.\n2) For the {level} added constraints, discriminate if the #Answer of Initial Instruction + {level} constraints# satisfies each constraint.\n3) In the final line, only output a Python LIST with {level} elements ('YES' or 'NO') indicating whether the answer satisfies each constraint.".format(level=level)

    return prompt



############################################################################################################################################## evaluation prompt for format constraints

def format_evaluation_prompt(evolve_instructions, answer):
    level = len(evolve_instructions)-1

    if level == 1:
        prompt = "Given an initial instruction, we add one format constraint and obtain the final instruction with 1 additional constraint.\n\n" \
                + "#Initial Instruction#\n" + evolve_instructions[0] + "\n\n"

        prompt += "#Initial Instruction + 1 constraint#\n" + evolve_instructions[1] + "\n\n"

        prompt += "#Answer of Initial Instruction + 1 constraint#\n" + answer + "\n\n"

        prompt += "#System#\n1) Please identify the 1 added constraint.\n2) Please descriminate if the #Answer of Initial Instruction + 1 constraint# satisfies the 1 added constraint.\n3) In the final line, only output a Python LIST with 1 element ('YES' or 'NO') indicating whether the answer satisfies the 1 added constraint."


    else:
        prompt = "Given an initial instruction, we add one format constraint per time and obtain the final instruction with {} additional constraints.\n\n".format(level) \
                + "#Initial Instruction#\n" + evolve_instructions[0] + "\n\n"

        prompt += "#Initial Instruction + 1 constraint#\n" + evolve_instructions[1] + "\n\n"

        for i in range(2, level+1):
            prompt += "#Initial Instruction + {} constraints#\n".format(i) + evolve_instructions[i] + "\n\n"

        prompt += "#Answer of Initial Instruction + {} constraints#\n".format(level) + answer + "\n\n"

        prompt += "#System#\n1) Please identify all {level} added constraints.\n2) For the {level} added constraints, discriminate if the #Answer of Initial Instruction + {level} constraints# satisfies each constraint.\n3) In the final line, only output a Python LIST with {level} elements ('YES' or 'NO') indicating whether the answer satisfies each constraint.".format(level=level)

    return prompt



############################################################################################################################################## evaluation prompt for mixed constraints

def mixed_evaluation_prompt(evolve_instructions, answer):
    level = len(evolve_instructions)-1

    if level == 1:
        prompt = "Given an initial instruction, we add one constraint and obtain the final instruction with 1 additional constraint.\n\n" \
                + "#Initial Instruction#\n" + evolve_instructions[0] + "\n\n"

        prompt += "#Initial Instruction + 1 constraint#\n" + evolve_instructions[1] + "\n\n"

        prompt += "#Answer of Initial Instruction + 1 constraint#\n" + answer + "\n\n"

        prompt += "#System#\n1) Please identify the 1 added constraint.\n2) Please descriminate if the #Answer of Initial Instruction + 1 constraint# satisfies the 1 added constraint.\n3) In the final line, only output a Python LIST with 1 element ('YES' or 'NO') indicating whether the answer satisfies the 1 added constraint."


    else:
        prompt = "Given an initial instruction, we add one constraint per time and obtain the final instruction with {} additional constraints.\n\n".format(level) \
                + "#Initial Instruction#\n" + evolve_instructions[0] + "\n\n"

        prompt += "#Initial Instruction + 1 constraint#\n" + evolve_instructions[1] + "\n\n"

        for i in range(2, level+1):
            prompt += "#Initial Instruction + {} constraints#\n".format(i) + evolve_instructions[i] + "\n\n"

        prompt += "#Answer of Initial Instruction + {} constraints#\n".format(level) + answer + "\n\n"

        prompt += "#System#\n1) Please identify all {level} added constraints.\n2) For the {level} added constraints, discriminate if the #Answer of Initial Instruction + {level} constraints# satisfies each constraint.\n3) In the final line, only output a Python LIST with {level} elements ('YES' or 'NO') indicating whether the answer satisfies each constraint.".format(level=level)

    return prompt



############################################################################################################################################## acquire_discriminative_eval_input 

def acquire_discriminative_eval_input(data_path, api_output_path, constraint_type, model_name, data_gpt4_discriminative_eval_input_path, gpt4_discriminative_eval_input_path):

    if not os.path.exists(data_gpt4_discriminative_eval_input_path):
        os.makedirs(data_gpt4_discriminative_eval_input_path)

    if not os.path.exists(gpt4_discriminative_eval_input_path):
        os.makedirs(gpt4_discriminative_eval_input_path)
    
    rule_based_source = ["E2E", "WIKIEVENTS", "CONLL2003", 
                        "text_editing", "cnn_dailymail", "xsum", "samsum", "gigaword", "arxiv", 
                        "BBH_logical", "BBH_time", "self_made_space", "gsm_8k"]

    data = data_match_api_output(data_path, api_output_path, constraint_type, model_name)

    with open(os.path.join(data_gpt4_discriminative_eval_input_path, "{0}_{1}_constraint.jsonl".format(model_name, constraint_type)), 'w', encoding='utf-8') as data_gpt_eval_input_file:
        with open(os.path.join(gpt4_discriminative_eval_input_path, "{0}_{1}_constraint.jsonl".format(model_name, constraint_type)), 'w', encoding='utf-8') as gpt_eval_input_file:
            
            evolve_instructions = []
            example_id = data[0]['example_id']

            for i in range(len(data)):
                
                if data[i]['example_id'] == example_id:
                    evolve_instructions.append(data[i]['instruction'])
                else:
                    example_id = data[i]['example_id']
                    evolve_instructions = [data[i]['instruction']]

                if data[i]['level'] > 0 and (data[i]['source'] not in rule_based_source):

                    if data[i]['category']!="format" or (data[i]['category']=="format" and data[i]['example_id'] not in [22, 30]):

                        data_gpt_eval_input_file.write(json.dumps({
                                                    "example_id": data[i]['example_id'],
                                                    "category": data[i]['category'],
                                                    "source": data[i]['source'],
                                                    "level": data[i]['level'],
                                                    'prompt_new': globals()[f"{constraint_type}_evaluation_prompt"](evolve_instructions, data[i]['generation'])
                                                    })+ "\n")

                        gpt_eval_input_file.write(json.dumps({
                                                    'prompt_new': globals()[f"{constraint_type}_evaluation_prompt"](evolve_instructions, data[i]['generation'])
                                                    })+ "\n")



############################################################################################################################################## HSR_SSR_evaluation

def paring_discriminative_generation(generation, level):
    try:
        satisify = generation.strip("```").strip().split('\n')[-1]

        if level == 1:
            if 'YES' in satisify:
                return 1, 1
            elif 'NO' in satisify:
                return 0, 0
            else:
                raise Exception('Invalid evaluation for level 1.')
        else:
            satisify_list = re.search(r'\[.*\]', satisify)
            if satisify_list:
                satisify_list = eval(satisify_list.group())
                if len(satisify_list) == level:
                    num_true = 0
                    for i in satisify_list:
                        if i == 'YES':
                            num_true += 1
                        elif i in ['NO', 'PARTIAL', 'MAYBE', 'UNKNOWN', 'N/A']:
                            num_true += 0
                        else:
                            raise Exception('Invalid element in the list.')
                    return int(num_true==level), num_true/level
                else:
                    raise Exception('Invalid number of elements in the list.')
            else:
                raise Exception('Invalid list that cannot be parsed.')

    except Exception as e:
        logger.error(f'{e}\nContent: {generation}\n'
                     'You must manually fix the evaluation.')
        return -1, -1



def discriminative_evaluation(data_gpt4_discriminative_eval_input_path, gpt4_discriminative_eval_output_path, constraint_type, model_name):
    data = []
    with open(os.path.join(data_gpt4_discriminative_eval_input_path, "{0}_{1}_constraint.jsonl".format(model_name, constraint_type)), 'r', encoding='utf-8') as data_file:
        for line in data_file:
            data.append(json.loads(line))

    output = []
    with open(os.path.join(gpt4_discriminative_eval_output_path, "{0}_{1}_constraint.jsonl".format(model_name, constraint_type)), 'r', encoding='utf-8') as output_file:
        for line in output_file:
            output.append(json.loads(line))

    # match
    for i in range(len(data)):
        for j in range(len(output)):
            if data[i]['prompt_new'] == output[j]['prompt_new']:
                data[i]['hard_satisfy'] = paring_discriminative_generation(output[j]['choices'][0]['message']['content'], data[i]['level'])[0]
                data[i]['soft_satisfy'] = paring_discriminative_generation(output[j]['choices'][0]['message']['content'], data[i]['level'])[1]
                break
            if j == len(output)-1 and data[i]['prompt_new'] != output[j]['prompt_new']:
                print(i)

    results = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    n_group = len(data) // 5

    for i in range(len(data)):
        if data[i]['level'] == 1:
            results[0][0] += data[i]['hard_satisfy']
            results[1][0] += data[i]['soft_satisfy']
        elif data[i]['level'] == 2:
            results[0][1] += data[i]['hard_satisfy']
            results[1][1] += data[i]['soft_satisfy']
        elif data[i]['level'] == 3:
            results[0][2] += data[i]['hard_satisfy']
            results[1][2] += data[i]['soft_satisfy']
        elif data[i]['level'] == 4:
            results[0][3] += data[i]['hard_satisfy']
            results[1][3] += data[i]['soft_satisfy']
        elif data[i]['level'] == 5:
            results[0][4] += data[i]['hard_satisfy']
            results[1][4] += data[i]['soft_satisfy']

    # print("\n[{0}: {1} result for {2} examples]".format(constraint_type, model_name, int(n_group)))
    # for i in range(len(results)):
    #     print("Level {}: {:.2f}%".format(i+1, results[i]/n_group*100))

    # return ["{:.2f}%".format(r/n_group*100) for r in results]

    return results, n_group



def save_discriminative_evaluation(data_path, api_output_path, data_gpt4_discriminative_eval_input_path, gpt4_discriminative_eval_output_path, constraint_type, model_names, evaluation_result_path):
    headers = ["model_name", "level 1", "level 2", "level 3", "level 4", "level 5"]
    
    ### compute HSR
    results = []
    for model_name in model_names:
        discriminative_result, discriminative_n_group = discriminative_evaluation(
                                            data_gpt4_discriminative_eval_input_path=data_gpt4_discriminative_eval_input_path, 
                                            gpt4_discriminative_eval_output_path=gpt4_discriminative_eval_output_path, 
                                            constraint_type=constraint_type, 
                                            model_name=model_name
                                            )

        rule_result, rule_n_group = rule_evaluation(
                                            data_path=data_path, 
                                            api_output_path=api_output_path, 
                                            constraint_type=constraint_type, 
                                            model_name=model_name
        )

        result = [i+j for i, j in zip(discriminative_result[0], rule_result)]
        n_group = discriminative_n_group + rule_n_group
        result = ["{:.2f}%".format(r/n_group*100) for r in result]

        results.append([model_name] + result)

    # plot
    x_labels = ['level 1', 'level 2', 'level 3', 'level 4', 'level 5']
    markers = ['o', 's', 'D', '*', '+', 'x', 'd', 'p', 'h'][:len(results)]

    fig, ax = plt.subplots()
    for (label, *values), marker in zip(results, markers):
        ax.plot(x_labels, [float(v.replace('%', '')) for v in values], label=label, marker=marker)
    ax.set_xlabel('Level')
    ax.set_ylabel('Hard Satisfaction Rate (%)')
    ax.set_title(f'Evaluation for {constraint_type} constraint')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_result_path, f"{constraint_type}_hsr.jpg"), format='jpg')


    with open(os.path.join(evaluation_result_path, f"{constraint_type}_hsr.csv"), mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(results)

    ### compute SSR
    results = []
    for model_name in model_names:
        discriminative_result, discriminative_n_group = discriminative_evaluation(
                                            data_gpt4_discriminative_eval_input_path=data_gpt4_discriminative_eval_input_path, 
                                            gpt4_discriminative_eval_output_path=gpt4_discriminative_eval_output_path, 
                                            constraint_type=constraint_type, 
                                            model_name=model_name
                                            )

        rule_result, rule_n_group = rule_evaluation(
                                            data_path=data_path, 
                                            api_output_path=api_output_path, 
                                            constraint_type=constraint_type, 
                                            model_name=model_name
        )

        result = [i+j for i, j in zip(discriminative_result[1], rule_result)]
        n_group = discriminative_n_group + rule_n_group
        result = ["{:.2f}%".format(r/n_group*100) for r in result]

        results.append([model_name] + result)

    # plot
    x_labels = ['level 1', 'level 2', 'level 3', 'level 4', 'level 5']
    markers = ['o', 's', 'D', '*', '+', 'x', 'd', 'p', 'h'][:len(results)]

    fig, ax = plt.subplots()
    for (label, *values), marker in zip(results, markers):
        ax.plot(x_labels, [float(v.replace('%', '')) for v in values], label=label, marker=marker)
    ax.set_xlabel('Level')
    ax.set_ylabel('Soft Satisfaction Rate (%)')
    ax.set_title(f'Evaluation for {constraint_type} constraint')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_result_path, f"{constraint_type}_ssr.jpg"), format='jpg')


    with open(os.path.join(evaluation_result_path, f"{constraint_type}_ssr.csv"), mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(results)



############################################################################################################################################## CSL_evaluation

def csl_evaluation(data_gpt4_discriminative_eval_input_path, gpt4_discriminative_eval_output_path, constraint_type, model_name):
    data = []
    with open(os.path.join(data_gpt4_discriminative_eval_input_path, "{0}_{1}_constraint.jsonl".format(model_name, constraint_type)), 'r', encoding='utf-8') as data_file:
        for line in data_file:
            data.append(json.loads(line))

    output = []
    with open(os.path.join(gpt4_discriminative_eval_output_path, "{0}_{1}_constraint.jsonl".format(model_name, constraint_type)), 'r', encoding='utf-8') as output_file:
        for line in output_file:
            output.append(json.loads(line))

    # match
    for i in range(len(data)):
        for j in range(len(output)):
            if data[i]['prompt_new'] == output[j]['prompt_new']:
                data[i]['hard_satisfy'] = paring_discriminative_generation(output[j]['choices'][0]['message']['content'], data[i]['level'])[0]
                data[i]['soft_satisfy'] = paring_discriminative_generation(output[j]['choices'][0]['message']['content'], data[i]['level'])[1]
                break
            if j == len(output)-1 and data[i]['prompt_new'] != output[j]['prompt_new']:
                print(i)

    all_consistency = 0
    n_group = 0

    for i in range(len(data)):
        if data[i]['level'] == 1:
            consistency = []
        consistency.append(data[i]['hard_satisfy'])

        if data[i]['level'] == 5:
            assert len(consistency) == 5
            count = 0
            for c in consistency:
                if c == 1:
                    count += 1
                else:
                    break
            all_consistency += count
            n_group += 1
        
    return all_consistency, n_group



def save_csl_evaluation(data_path, api_output_path, data_gpt4_discriminative_eval_input_path, gpt4_discriminative_eval_output_path, constraint_type, model_names, evaluation_result_path):
    headers = ["model_name", "CSL"]
    results = []

    for model_name in model_names:
        discriminative_consistency, discriminative_n_group = csl_evaluation(
                                            data_gpt4_discriminative_eval_input_path=data_gpt4_discriminative_eval_input_path, 
                                            gpt4_discriminative_eval_output_path=gpt4_discriminative_eval_output_path, 
                                            constraint_type=constraint_type, 
                                            model_name=model_name
                                            )

        rule_consistency, rule_n_group = csl_five_constraint(
                                            data_path=data_path, 
                                            api_output_path=api_output_path, 
                                            constraint_type=constraint_type, 
                                            model_name=model_name
        )

        n_group = discriminative_n_group + rule_n_group
        csl = round((discriminative_consistency + rule_consistency) / n_group, 1)
        results.append([model_name, csl])

    with open(os.path.join(evaluation_result_path, f"{constraint_type}_csl.csv"), mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(results)



############################################################################################################################################## compute failure consitency

def is_1_after_0(lst):
    if len(lst) != 5:
        raise ValueError("THE LIST SHOULD HAVE 5 ELEMENTS")
    for i in range(len(lst)):
        if lst[i] == 0:
            for j in range(i+1, len(lst)):
                if lst[j] == 1:
                    return True
    return False


def compute_consistency(data_path, api_output_path, data_gpt4_discriminative_eval_input_path, gpt4_discriminative_eval_output_path, constraint_type, model_name):
    
    if constraint_type != 'example':
    
        data = []
        with open(os.path.join(data_gpt4_discriminative_eval_input_path, "{0}_{1}_constraint.jsonl".format(model_name, constraint_type)), 'r', encoding='utf-8') as data_file:
            for line in data_file:
                data.append(json.loads(line))

        output = []
        with open(os.path.join(gpt4_discriminative_eval_output_path, "{0}_{1}_constraint.jsonl".format(model_name, constraint_type)), 'r', encoding='utf-8') as output_file:
            for line in output_file:
                output.append(json.loads(line))

        # match
        for i in range(len(data)):
            for j in range(len(output)):
                if data[i]['prompt_new'] == output[j]['prompt_new']:
                    data[i]['hard_satisfy'] = paring_discriminative_generation(output[j]['choices'][0]['message']['content'], data[i]['level'])[0]
                    break
                if j == len(output)-1 and data[i]['prompt_new'] != output[j]['prompt_new']:
                    print(i)

        n_consistency = 0
        n_group = 0
        results = []

        for i in range(len(data)):

            if data[i]['level'] == 1:
                results = []

            results.append(data[i]['hard_satisfy'])

            if data[i]['level'] == 5:
                if 0 in results[:4]:
                    n_group += 1
                    n_consistency += (not is_1_after_0(results))
        
        return n_consistency, n_group

    else:
        with open(os.path.join(data_path, 'example_constraint_examples.json'), 'r', encoding='utf-8') as data_file:
            data = json.load(data_file)

        output = []
        with open(os.path.join(api_output_path, '{}_example_constraint.jsonl'.format(model_name)), 'r', encoding='utf-8') as output_file:
            for line in output_file:
                output.append(json.loads(line))

        # match
        for i in range(len(data)):
            for j in range(len(output)):
                if data[i]['instruction'] == output[j]['prompt_new']:
                    data[i]['generation'] = output[j]['choices'][0]['message']['content']
                    break
                if j == len(output)-1 and data[i]['instruction'] != output[j]['prompt_new']:
                    print(i)
        
        # check correct
        for i in range(len(data)):
            template = data[i]['target'].replace('{instruction}\n', '')
            generation = data[i]['generation']
            data[i]['correct'] = check_match(template, generation)

        n_consistency = 0
        n_group = 0
        results = []

        for i in range(len(data)):

            if data[i]['level'] == 1:
                results = []

            results.append(data[i]['correct'])

            if data[i]['level'] == 5:
                if 0 in results[:4]:
                    n_group += 1
                    n_consistency += (not is_1_after_0(results))

        
        return n_consistency, n_group
