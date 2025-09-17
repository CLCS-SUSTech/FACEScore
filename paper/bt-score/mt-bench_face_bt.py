from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import pyarrow as pa
import pandas as pd
import numpy as np
import argparse
import torch
import os

from face import spectrum_pipeline, evaluate_pipeline
from inference import inference


# sys envs
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3, 4'
os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'


# arg parse
parser = argparse.ArgumentParser()
parser.add_argument('--zs', default=False, action='store_true')
parser.add_argument('--log', default=False, action='store_true')
parser.add_argument('--real', default=False, action='store_true')
parser.add_argument('--est_path', type=str, required=True)
args = parser.parse_args()

# ref: https://datascience.oneoffcoder.com/btl-model.html
def get_estimate(i, p, df):
    get_prob = lambda i, j: np.nan if i == j else p.iloc[i] + p.iloc[j]
    n = df.iloc[i].sum()

    d_n = df.iloc[i] + df.iloc[:, i]
    d_d = pd.Series([get_prob(i, j) for j in range(len(p))], index=p.index)
    d = (d_n / d_d).sum()

    return n / d

def estimate_p(p, df):
    return pd.Series([get_estimate(i, p, df) for i in range(df.shape[0])], index=p.index)

def iterate(df, p=None, n=20, sorted=True):
    if p is None:
        p = pd.Series([1 for _ in range(df.shape[0])], index=list(df.columns))

    estimates = [p]

    for _ in range(n):
        p = estimate_p(p, df)
        p = p / p.sum()
        estimates.append(p)

    p = p.sort_values(ascending=False) if sorted else p
    return p, pd.DataFrame(estimates)



est_path = args.est_path
mt_bench = load_dataset('/nas/data/mt-bench', trust_remote_code=True)
models = np.unique(mt_bench['train']['model_a'])
inference_model_name = est_path.split('/')[-1]
gpu_mem = 0.9

metrics = ['so', 'corr', 'spear', 'emd', 'kl', 'js', 'surprisal']

schema = pa.schema({
    'input': pa.string(),
    'output': pa.string()
})

llm = LLM(
    est_path,
    gpu_memory_utilization=0.8,
    max_model_len=2048,
    tensor_parallel_size=torch.cuda.device_count()
)
sampling_params = SamplingParams(
    temperature=0, 
    prompt_logprobs=0, 
    max_tokens=1
)
tokenizer = AutoTokenizer.from_pretrained(est_path)

win_counts = {
    'so': {model_a: {model_b: 0 for model_b in models} for model_a in models},
    'corr': {model_a: {model_b: 0 for model_b in models} for model_a in models},
    'spear': {model_a: {model_b: 0 for model_b in models} for model_a in models},
    'emd': {model_a: {model_b: 0 for model_b in models} for model_a in models},
    'kl': {model_a: {model_b: 0 for model_b in models} for model_a in models},
    'js': {model_a: {model_b: 0 for model_b in models} for model_a in models},
    'surprisal': {model_a: {model_b: 0 for model_b in models} for model_a in models}
}
line_results = []

# GPT4 records as groundtruth
gpt4_dict = {}
for line in tqdm(mt_bench['train']):
    if line['model_a'] == 'gpt-4':
        for text in line['conversation_a']:
            if text['role'] == 'user':
                line_input = text['content']
            else:
                if line_input in gpt4_dict.keys():
                    if not (text['content'] in gpt4_dict[line_input]):
                        gpt4_dict[line_input].append(text['content'])
                else:
                    gpt4_dict[line_input] = [text['content']]
    elif line['model_b'] == 'gpt-4':
        for text in line['conversation_b']:
            if text['role'] == 'user':
                line_input = text['content']
            else:
                if line_input in gpt4_dict.keys():
                    if not (text['content'] in gpt4_dict[line_input]):
                        gpt4_dict[line_input].append(text['content'])
                else:
                    gpt4_dict[line_input] = [text['content']]


for line in tqdm(mt_bench['train']):
    def eval_line(conversation_name):
        input = []
        output = []
        gpt4_output = []
        for text in line[conversation_name]:
            if text['role'] == 'user':
                text_input = text['content']
            else:
                if text_input != '' and text['content'] != '':
                    input.extend([text_input] * len(gpt4_dict[text_input]))
                    output.extend([text['content']] * len(gpt4_dict[text_input]))
                    gpt4_output.extend(gpt4_dict[text_input])
        data = pa.Table.from_pydict(
            dict(
                zip(schema.names, (input, output))
            ),
            schema=schema
        )
        data = Dataset(data)
        # model spectrum
        inferenced_data_model = inference(llm, sampling_params, data, tokenizer, False)
        model_spectrum = spectrum_pipeline(inferenced_data_model['logprobs'], args)

        data = pa.Table.from_pydict(
            dict(
                zip(schema.names, (input, gpt4_output))
            ),
            schema=schema
        )
        data = Dataset(data)
        # groundtruth spectrum
        inferenced_data_human = inference(llm, sampling_params, data, tokenizer, False)
        human_spectrum = spectrum_pipeline(inferenced_data_human['logprobs'], args)
        raw_results = evaluate_pipeline(human_spectrum, model_spectrum, metrics)
        for idx, logprob in enumerate(inferenced_data_model['logprobs']):
            raw_results[idx]['surprisal'] = np.mean(logprob)
        return_results = []
        # complicated handling for multi-groundtruth
        # turned out useless
        last_input = ''
        for idx, line_result in enumerate(raw_results):
            line_input = input[idx]
            if line_input != last_input:
                if last_input != '':
                    line_return_result = {}
                    for metric in metrics:
                        if metric == 'surprisal':
                            line_return_result[metric] = group_results[metric][0]
                        else:
                            line_return_result[metric] = np.max(group_results[metric]) if metric in ['so', 'corr', 'spear'] else np.min(group_results[metric])
                    return_results.append({'input': input[idx-1], 'output': output[idx-1], 'gpt4_output': gpt4_output[idx-1], 'result': line_return_result})
                group_results = {metric: [] for metric in metrics}
                last_input = line_input
            for metric in metrics:
                group_results[metric].append(line_result[metric])
        line_return_result = {}
        for metric in metrics:
            if metric == 'surprisal':
                line_return_result[metric] = group_results[metric][0]
            else:
                line_return_result[metric] = np.max(group_results[metric]) if metric in ['so', 'corr', 'spear'] else np.min(group_results[metric])
        return_results.append({'input': line_input, 'output': output[idx], 'gpt4_output': gpt4_output[idx], 'result': line_return_result})

        return return_results

    # model a eval
    result_a = eval_line('conversation_a')
    # model b eval
    result_b = eval_line('conversation_b')
    for line_result_a, line_result_b in zip(result_a, result_b):
        # construct line record
        line_results.append({'model_a': line['model_a'], 'model_b': line['model_b'], 'input': line_result_a['input'], 'output_a': line_result_a['output'], 'output_b': line_result_b['output'], 'gpt4_output': line_result_a['gpt4_output'], 'face_a': line_result_a['result'], 'face_b': line_result_b['result']})
    for metric in metrics:

        for line_result_a, line_result_b in zip(result_a, result_b):
            # greater better
            mean_result_a = line_result_a['result'][metric]
            mean_result_b = line_result_b['result'][metric]
            if metric in ['so', 'corr', 'spear']:
                if mean_result_a > mean_result_b:
                    win_counts[metric][line['model_b']][line['model_a']] += 1
                elif mean_result_a < mean_result_b:
                    win_counts[metric][line['model_a']][line['model_b']] += 1
            else:
                if mean_result_a < mean_result_b:
                    win_counts[metric][line['model_b']][line['model_a']] += 1
                elif mean_result_a > mean_result_b:
                    win_counts[metric][line['model_a']][line['model_b']] += 1

result_dict = {}
for metric, value in win_counts.items():
    print(metric)
    print('-' * 10)
    df = pd.DataFrame(value)
    p, estimates = iterate(df, n=100)
    print(p)
    print('=' * 10)
    result_dict[metric] = {}
    for model, model_p in p.items():
        result_dict[metric][model] = model_p


# save records
record_data = Dataset.from_list(line_results)
record_data.save_to_disk(f'./mt-bench-bt/face_est-{inference_model_name}_{'zscore' if args.zs else 'raw'}{'_real' if args.real else ''}_records')

# save results
result_path = f'./mt-bench-bt/face_est-{inference_model_name}_{'zscore' if args.zs else 'raw'}{'_real' if args.real else ''}_result.csv'
with open(result_path, 'w') as f:
    f.write(',')
    for metric in metrics:
        f.write(metric + ',')
    f.write('\n')
    for model in models:
        f.write(model + ',')
        for metric in metrics:
            f.write(str(result_dict[metric][model]) + ',')
        f.write('\n')