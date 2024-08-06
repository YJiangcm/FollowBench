from datasets import load_dataset
from promptsource.templates import DatasetTemplates
import random
import json
from tqdm import tqdm


all_a_templates = [
    "{instruction}\n{answer}",
    "{instruction}\nA: {answer}",
    "{instruction}\nAnswer: {answer}",
    "{instruction}\nANSWER: {answer}",
    "{instruction}\n[Answer]\n{answer}",
    "{instruction}\n#Answer#\n{answer}",
    "{instruction}\nThe answer is: {answer}",
    "{instruction}\n{{'answer': '{answer}'}}",
    "{instruction}\n{{'Answer': '{answer}'}}",
    "{instruction}\n<body>{answer}</body>",
    "{instruction}\nResponse: {answer}",
    "{instruction}\nRESPONSE: {answer}",
    "{instruction}\n[Response]\n{answer}",
    "{instruction}\n#Response#\n{answer}",
    "{instruction}\nThe response is: {answer}",
    "{instruction}\n{{'response': '{answer}'}}",
    "{instruction}\n{{'Response': '{answer}'}}",
    "{instruction}\nBot: {answer}",
    "{instruction}\nBOT: {answer}",
    "{instruction}\n[Bot]\n{answer}",
    "{instruction}\n#Bot#\n{answer}",
    "{instruction}\nThe response of the bot is: {answer}",
    "{instruction}\n{{'bot': '{answer}'}}",
    "{instruction}\n{{'Bot': '{answer}'}}",
    "{instruction}\nAI assistant: {answer}",
    "{instruction}\n[AI assistant]\n{answer}",
    "{instruction}\n#AI assistant#\n{answer}",
    "{instruction}\nThe response of the AI assistant is: {answer}",
    "{instruction}\n{{'AI assistant': '{answer}'}}"
]


def split_integer(m, n):
    if n > m:
        print("Error: n should be less than or equal to m")
        return None

    # 生成n-1个随机整数
    random_numbers = sorted(random.sample(range(1, m), n - 1))
    
    # 添加0和m作为端点
    random_numbers = [0] + random_numbers + [m]
    
    # 计算相邻数字之间的差值
    result = [random_numbers[i+1] - random_numbers[i] for i in range(n)]
    
    return result


def create_example_constraint(all_q_templates, sampled_dataset, sampled_q_templates, sampled_a_templates, n_shot, n_constraint):

    selected_q_templates = sampled_q_templates[: n_constraint]
    match_q_template = selected_q_templates[0]

    selected_a_templates = sampled_a_templates[: n_constraint]
    match_a_template = selected_a_templates[0]


    split_integers = split_integer(n_shot, n_constraint)
    five_q_templates = []
    five_a_templates = []
    
    for i in range(len(split_integers)):
        for _ in range(split_integers[i]):
            five_q_templates.append(selected_q_templates[i]) # a, b, b, c, c
            five_a_templates.append(selected_a_templates[i]) # a, b, b, c, c

    five_q_a_templates = [{"q_template": q, "a_template": a} for q, a in zip(five_q_templates, five_a_templates)]
    random.shuffle(five_q_a_templates)
    data = [{'data': d, "q_template": f['q_template'], "a_template": f['a_template']} for d, f in zip(sampled_dataset, five_q_a_templates)]


    few_shot = ""
    for d in data:
        instruction, answer = all_q_templates[d['q_template']].apply(d['data'])
        few_shot += d['a_template'].format(instruction=instruction, answer=answer)
        few_shot += "\n\n"
    instruction, answer = all_q_templates[match_q_template].apply(sampled_dataset[-1])
    few_shot += instruction

    answer = match_a_template.replace('{instruction}\n', '').format(answer=answer)

    return few_shot, match_a_template, answer


dataset_names = [
    "aeslc", 
    "ag_news", 
    "art", 
    "billsum",
    "bing_coronavirus_query_set",
    "biosses",
    "common_gen",
    "commonsense_qa",
    "craigslist_bargains",
    "dream",
    "drop",
    "emotion",
    "freebase_qa",
    "generated_reviews_enth",
    "google_wellformed_query",
    "gutenberg_time",
    "hans", 
    "health_fact",
    "imdb",
    "lambada",
    "limit",
    "math_qa",
    "mwsc",
    "narrativeqa",
    "newspop",
    "numer_sense",
    "onestop_english",
    "piqa",
    "poem_sentiment",
    "qa_srl",
    "quoref",
    "rotten_tomatoes",
    "samsum",
    "sciq",
    "species_800",
    "squad",
    "web_questions",
    "wiqa",
    "xsum", 
    "zest", 
    ]

n_sample = 1
n_shot = 5
samples = []

for i in tqdm(range(len(dataset_names))):
    print(dataset_names[i])
    dataset = load_dataset(dataset_names[i], split="train")
    all_q_templates = DatasetTemplates(dataset_names[i])
    assert(len(all_q_templates) >= n_shot)

    for j in range(n_sample):
        shuffled_dataset = dataset.shuffle()
        sampled_dataset = shuffled_dataset.select(range(n_shot+1))
        sampled_q_templates = random.sample(all_q_templates.name_to_id_mapping.keys(), n_shot)
        sampled_a_templates = random.sample(all_a_templates, n_shot)
        
        for level in range(n_shot):
            few_shot, match_a_template, answer = create_example_constraint(all_q_templates, sampled_dataset, sampled_q_templates, sampled_a_templates, n_shot=n_shot, n_constraint=level+1)
            samples.append({
                "example_id": i + j + 1,
                "category": "example",
                "source": dataset_names[i],
                "level": level+1,
                "instruction": few_shot,
                "target": match_a_template,
                "answer": answer
            })

print(f"create {len(samples)} samples.")
json_str = json.dumps(samples, indent=4)
with open('example_constraints.json', mode='w', encoding='utf-8') as json_file:
    json_file.write(json_str)
