# FollowBench
FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models


## Data
The data of FollowBench can be found in the [data/](data/).


## How to Implement

### Install Dependencies

```
conda create -n followbench python=3.10
conda activate followbench
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Model Inference
```bash
cd FollowBench/
python code/model_inference.py --model-path <model_name_or_path>
```

### Evaluation
You should first use GPT-4's API to acquire the LLM-based evaluation results, then we can organize and merge the rule-based evaluation results and LLM-based evaluation results using the following script:
```bash
cd FollowBench/
python code/eval.py --model_names <a_list_of_evaluated_models>
```



## Citation
Please cite our paper if you use the code in this repo.
```
@misc{jiang2023followbench,
      title={FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models}, 
      author={Yuxin Jiang and Yufei Wang and Xingshan Zeng and Wanjun Zhong and Liangyou Li and Fei Mi and Lifeng Shang and Xin Jiang and Qun Liu and Wei Wang},
      year={2023},
      eprint={2310.20410},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
