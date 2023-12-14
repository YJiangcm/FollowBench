import json
import csv
import os
import re
import ast
import matplotlib.pyplot as plt
import string

from utils import data_match_api_output_1



def rule_evaluation(data_path, api_output_path, constraint_type, model_name):
    
    rule_based_source = ["E2E", "WIKIEVENTS", "CONLL2003", 
                        "text_editing", "cnn_dailymail", "xsum", "samsum", "gigaword", "arxiv", 
                        "BBH_logical", "BBH_time", "self_made_space", "gsm_8k"]

    data = data_match_api_output_1(data_path, api_output_path, constraint_type, model_name)

    results = [0, 0, 0, 0, 0]
    n_group = 0

    for i in range(len(data)):
        if data[i]['level'] > 0 and data[i]['source'] in rule_based_source:
            n_group += 1
            source = data[i]['source']
            results[data[i]['level']-1] += globals()[f"rule_evaluation_{source}"](data[i]['generation'].encode('utf-8').decode('utf-8'), data[i]['target'].encode('utf-8').decode('utf-8'), data[i]['level'])
        
        if data[i]['level'] > 0 and data[i]['category'] == "format" and data[i]['example_id'] in [22]:
            n_group += 1
            results[data[i]['level']-1] += rule_evaluation_format(data[i]['generation'].encode('utf-8').decode('utf-8'), data[i]['example_id'], data[i]['level'])
    
    n_group = n_group // 5

    return results, n_group



###########################################################################################################
def contain_word(generation, word):
    pattern = re.compile(word)
    return bool(re.search(pattern, generation))


def count_sentences(generation):
    sentences = re.split(r'[。！？]', generation)
    sentences = [s for s in sentences if s]
    return len(sentences)


def count_sentence_words_less(generation, num):
    sentences = re.split(r'[。！？]', generation)
    sentences = [s for s in sentences if s]
    for s in sentences:
        if len(s) >= num:
            return False
    return True


def count_sentence_words_more(generation, num):
    sentences = re.split(r'[。！？]', generation)
    sentences = [s for s in sentences if s]
    for s in sentences:
        if len(s) <= num:
            return False
    return True


def n_sentence_contain_word(generation, num, word):
    sentences = re.split(r'[。！？]', generation)
    sentences = [s for s in sentences if s]
    if len(sentences) < num:
        return False
    return contain_word(sentences[num-1], word)


def contains_arabic_numerals(generation):
    for char in generation:
        if char.isnumeric():
            return True
    return False


def paragraphs_start_with_number_marker(s):
    paragraphs = s.split('\n\n')

    for paragraph in paragraphs:
        paragraph = paragraph.lstrip()
        if not paragraph:
            continue
        if not re.match(r'\d+\.', paragraph):
            return False

    return True
###########################################################################################################


def rule_evaluation_E2E(generation, target, level):
    return generation == target


def rule_evaluation_WIKIEVENTS(generation, target, level):
    generation = generation.split('\n')
    target = target.split('\n')
    for g in generation:
        if g not in target:
            return False
    return True


def rule_evaluation_CONLL2003(generation, target, level):
    return generation in ast.literal_eval(target)


def rule_evaluation_text_editing(generation, target, level):
    return generation == target


def rule_evaluation_cnn_dailymail(generation, target, level):
    if level == 1:
        return count_sentences(generation) == 3
    elif level == 2:
        return count_sentences(generation) == 3 and count_sentence_words_less(generation, 15)
    elif level == 3:
        return count_sentences(generation) == 3 and count_sentence_words_less(generation, 15) and (n_sentence_contain_word(generation, 1, '波特') and n_sentence_contain_word(generation, 2, '演员') and n_sentence_contain_word(generation, 3, '电影'))
    elif level == 4:
        return count_sentences(generation) == 3 and count_sentence_words_less(generation, 15) and (n_sentence_contain_word(generation, 1, '波特') and n_sentence_contain_word(generation, 2, '演员') and n_sentence_contain_word(generation, 3, '电影')) and not n_sentence_contain_word(generation, 2, '奢侈')
    else:
        return count_sentences(generation) == 3 and count_sentence_words_less(generation, 15) and (n_sentence_contain_word(generation, 1, '波特') and n_sentence_contain_word(generation, 2, '演员') and n_sentence_contain_word(generation, 3, '电影')) and not n_sentence_contain_word(generation, 2, '奢侈') and n_sentence_contain_word(generation, 3, '了')


def rule_evaluation_xsum(generation, target, level):
    if level == 1:
        return count_sentences(generation) == 1
    elif level == 2:
        return count_sentences(generation) == 1 and len(generation) < 20
    elif level == 3:
        return count_sentences(generation) == 1 and len(generation) < 20 and not contain_word(generation, '牛顿斯图尔特')
    elif level == 4:
        return count_sentences(generation) == 1 and len(generation) < 20 and not contain_word(generation, '牛顿斯图尔特') and contain_word(generation, '在')
    else:
        return count_sentences(generation) == 1 and len(generation) < 20 and not contain_word(generation, '牛顿斯图尔特') and contain_word(generation, '在') and contain_word(generation, '操作')


def rule_evaluation_samsum(generation, target, level):
    if level == 1:
        return count_sentences(generation) == 1
    elif level == 2:
        return count_sentences(generation) == 1 and len(generation) < 15
    elif level == 3:
        return count_sentences(generation) == 1 and len(generation) < 15 and contain_word(generation, '琐碎的')
    elif level == 4:
        return count_sentences(generation) == 1 and len(generation) < 15 and contain_word(generation, '琐碎的') and not contain_word(generation, '拖沓的')
    else:
        return count_sentences(generation) == 1 and len(generation) < 15 and contain_word(generation, '琐碎的') and not contain_word(generation, '拖沓的') and sum(1 for char in generation if char in (string.punctuation + "，。：’“”")) == 1


def rule_evaluation_gigaword(generation, target, level):
    if level == 1:
        return len(generation) == 8
    elif level == 2:
        return len(generation) == 8 and not any(char in (string.punctuation + "，。：’“”") for char in generation)
    elif level == 3:
        return len(generation) == 8 and not any(char in (string.punctuation + "，。：’“”") for char in generation) and not contains_arabic_numerals(generation)
    elif level == 4:
        return len(generation) == 8 and not any(char in (string.punctuation + "，。：’“”") for char in generation) and not contains_arabic_numerals(generation) and not contain_word(generation, '公交车')
    else:
        return len(generation) == 8 and not any(char in (string.punctuation + "，。：’“”") for char in generation) and not contains_arabic_numerals(generation) and not contain_word(generation, '公交车') and contain_word(generation, '在')


def rule_evaluation_arxiv(generation, target, level):
    if level == 1:
        return count_sentences(generation) == 1
    elif level == 2:
        return count_sentences(generation) == 1 and len(generation) <= 20
    elif level == 3:
        return count_sentences(generation) == 1 and len(generation) <= 20 and generation[:2] == "我们"
    elif level == 4:
        return count_sentences(generation) == 1 and len(generation) <= 20 and generation[:2] == "我们" and contain_word(generation, '激活')
    else:
        return count_sentences(generation) == 1 and len(generation) <= 20 and generation[:2] == "我们" and contain_word(generation, '激活') and not contain_word(generation, 'transformer')


def rule_evaluation_BBH_logical(generation, target, level):
    match = re.findall(r'\([A-Z]\)', generation)
    if match:
        answer = match[-1]  # Getting the last occurrence
        return answer == target
    else:
        return False


def rule_evaluation_BBH_time(generation, target, level):
    match = re.findall(r"\d{2}/\d{2}/\d{4}", generation)
    if match:
        answer = match[-1]  # Getting the last occurrence
        return answer == target
    else:
        return False


def rule_evaluation_self_made_space(generation, target, level):
    return target in generation


def rule_evaluation_gsm_8k(generation, target, level):
    match = re.findall(r'\d+美元', generation)
    if match:
        answer = match[-1]  # Getting the last occurrence
        return answer == target
    else:
        return False


def rule_evaluation_format(generation, example_id, level):
    if example_id == 22:
        if level == 1:
            return len(generation.split('\n\n')) == 3
        elif level == 2:
            return len(generation.split('\n\n')) == 3 and (count_sentences(generation.split('\n\n')[0]) < 3 and count_sentences(generation.split('\n\n')[1]) < 3 and count_sentences(generation.split('\n\n')[2]) < 3)
        elif level == 3:
            return len(generation.split('\n\n')) == 3 and (count_sentences(generation.split('\n\n')[0]) < 3 and count_sentences(generation.split('\n\n')[1]) < 3 and count_sentences(generation.split('\n\n')[2]) < 3) and count_sentence_words_more(generation, 20)
        elif level == 4:
            return len(generation.split('\n\n')) == 3 and (count_sentences(generation.split('\n\n')[0]) < 3 and count_sentences(generation.split('\n\n')[1]) < 3 and count_sentences(generation.split('\n\n')[2]) < 3) and count_sentence_words_more(generation, 20) and paragraphs_start_with_number_marker(generation)
        else:
            return len(generation.split('\n\n')) == 4 and (count_sentences(generation.split('\n\n')[0]) < 3 and count_sentences(generation.split('\n\n')[1]) < 3 and count_sentences(generation.split('\n\n')[2]) < 3) and count_sentence_words_more(generation, 20) and paragraphs_start_with_number_marker(generation) and generation[-len("这些是建议"):] == "这些是建议"


############################################################################################################################################## example constraint evaluation

def check_match(target, generation):
    pattern_str = target.replace('{{', '{').replace('}}', '}').replace('{answer}', '.*')
    pattern_str = re.escape(pattern_str).replace('\\.\\*', '.*')
    match = re.fullmatch(pattern_str, generation)
    return bool(match)


def evaluate_example_constraint(data_path, api_output_path, model_name):

    with open(os.path.join(data_path, 'example_constraints.json'), 'r', encoding='utf-8') as data_file:
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
                data[i]['generation'] = ""
    
    # check correct
    for i in range(len(data)):
        template = data[i]['target'].replace('{instruction}\n', '')
        generation = data[i]['generation']
        data[i]['correct'] = check_match(template, generation)

    
    results = [0, 0, 0, 0, 0]
    n_group = len(data) // 5

    for i in range(len(data)):
        if data[i]['level'] == 1:
            results[0] += data[i]['correct']
        elif data[i]['level'] == 2:
            results[1] += data[i]['correct']
        elif data[i]['level'] == 3:
            results[2] += data[i]['correct']
        elif data[i]['level'] == 4:
            results[3] += data[i]['correct']
        elif data[i]['level'] == 5:
            results[4] += data[i]['correct']

    # print("\n[example: {0} result for {1} examples]".format(model_name, int(n_group)))
    # for i in range(len(results)):
    #     print("Level {}: {:.2f}%".format(i+1, results[i]/n_group*100))

    return ["{:.2f}%".format(r/n_group*100) for r in results]


def save_evaluate_example_constraint(data_path, api_output_path, model_names, evaluation_result_path):
    headers = ["model_name", "level 1", "level 2", "level 3", "level 4", "level 5"]
    results = []

    for model_name in model_names:
        result = evaluate_example_constraint(
                                            data_path=data_path, 
                                            api_output_path=api_output_path, 
                                            model_name=model_name
                                            )
        results.append([model_name] + result)


    ### plot
    x_labels = ['level 1', 'level 2', 'level 3', 'level 4', 'level 5']
    markers = ['o', 's', 'D', '*', '+', 'x', 'd', 'p', 'h'][:len(results)]

    fig, ax = plt.subplots()
    for (label, *values), marker in zip(results, markers):
        ax.plot(x_labels, [float(v.replace('%', '')) for v in values], label=label, marker=marker)
    ax.set_xlabel('Level')
    ax.set_ylabel('Hard Satisfaction Rate (%)')
    ax.set_title(f'Evaluation for example constraint')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_result_path, "example_rule_based.jpg"), format='jpg')


    with open(os.path.join(evaluation_result_path, "example_rule_based.csv"), mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(results)


############################################################################################################################################## CSL_evaluation

def csl_five_constraint(data_path, api_output_path, constraint_type, model_name):
    
    rule_based_source = ["E2E", "WIKIEVENTS", "CONLL2003", 
                        "text_editing", "cnn_dailymail", "xsum", "samsum", "gigaword", "arxiv", 
                        "BBH_logical", "BBH_time", "self_made_space", "gsm_8k"]

    data = data_match_api_output_1(data_path, api_output_path, constraint_type, model_name)

    all_consistency = 0
    n_group = 0

    for i in range(len(data)):
        if data[i]['level'] > 0 and data[i]['source'] in rule_based_source:

            if data[i]['level'] == 1:
                consistency = []
            
            source = data[i]['source']
            satisfy = globals()[f"rule_evaluation_{source}"](data[i]['generation'].encode('utf-8').decode('utf-8'), data[i]['target'].encode('utf-8').decode('utf-8'), data[i]['level'])
            consistency.append(satisfy)

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
        
        if data[i]['level'] > 0 and data[i]['category'] == "format" and data[i]['example_id'] in [22]:

            if data[i]['level'] == 1:
                consistency = []
            
            satisfy = rule_evaluation_format(data[i]['generation'].encode('utf-8').decode('utf-8'), data[i]['example_id'], data[i]['level'])
            consistency.append(satisfy)

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



def csl_example_constraint(data_path, api_output_path, model_name):

    with open(os.path.join(data_path, 'example_constraints.json'), 'r', encoding='utf-8') as data_file:
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
                data[i]['generation'] = ""
    
    # check correct
    for i in range(len(data)):
        template = data[i]['target'].replace('{instruction}\n', '')
        generation = data[i]['generation']
        data[i]['correct'] = check_match(template, generation)

    all_consistency = 0
    n_group = 0

    for i in range(len(data)):

        if data[i]['level'] == 1: # 12345 12345
            consistency = []

        consistency.append(data[i]['correct'])

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



def save_csl_example_constraint(data_path, api_output_path, model_names, evaluation_result_path):
    headers = ["model_name", "CSL"]
    results = []

    for model_name in model_names:
        rule_consistency, n_group = csl_example_constraint(
                                            data_path=data_path, 
                                            api_output_path=api_output_path, 
                                            model_name=model_name
                                            )
        csl = round(rule_consistency/n_group, 1)
        results.append([model_name, csl])

    with open(os.path.join(evaluation_result_path, "example_csl.csv"), mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(results)