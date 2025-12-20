from submodules.mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
import re

from utils import *



def find_five_digit_number(input_string):
    # Use regex to search for a 5-digit number
    match = re.search(r'\b\d{5}\b', input_string)
    if match:
        return match.group(0)
    else:
        return None

def find_subsequence_index(lst, subsequence):
    subseq_len = len(subsequence)
    for i in range(len(lst) - subseq_len + 1):
        if lst[i:i + subseq_len] == subsequence:
            return i
    return -1 

def parse_tokens(token_seq, tokenizer, wanted_key, wanted_value, i_query):
    parsed_info = {}
    
    wanted_key_toks = tokenizer.encode(wanted_key) if i_query == 0 else tokenizer.encode(f' {wanted_key}')
    wanted_value_toks = tokenizer.encode(wanted_value)

    decoded_seq = [tokenizer.decode([token_seq[i]]) for i in range(len(token_seq))]
    token_seq = token_seq.cpu().tolist()

    parsed_info['i_query'] = i_query
    parsed_info['per_token_decoding'] = [x for x in zip(token_seq, decoded_seq)]
    try:
        end_of_text_idx = token_seq.index(11) # beyond this token we have garbage
    except:
        end_of_text_idx = -1 # sequence terminated at max len
    parsed_info['end_of_text_idx'] = end_of_text_idx
    parsed_info['key_in_query_idx'] = token_seq.index(10) - 3 # we find the end of the query, the result is 3 token before that
    parsed_info['query_start_idx'] = find_subsequence_index(token_seq, [193, 5000, 2615, 248, 3312, 1386,25])
    parsed_info['query_end_idx'] = token_seq.index(10)
    parsed_info['key_in_context_idx'] = find_subsequence_index(token_seq, wanted_key_toks)
    parsed_info['value_in_context_idx'] = find_subsequence_index(token_seq, wanted_value_toks)
    wanted_val_idx_in_ans = find_subsequence_index(token_seq[parsed_info['query_end_idx']:parsed_info['end_of_text_idx']], wanted_value_toks)
    parsed_info['value_in_answer_idx'] = wanted_val_idx_in_ans + parsed_info['query_end_idx'] if wanted_val_idx_in_ans >= 0 else wanted_val_idx_in_ans
    
    return parsed_info


def run_ar_experiment(config, start_datetime_str):

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model_name = config['model_name']
    mamba = MambaLMHeadModel.from_pretrained(model_name, cache_dir=config['cache_dir']).to(torch.float16).to(config["device"])
    
    seeds = [123,234,345,456,567]
    num_of_facts_to_test = [10,30,50,70,90,110,130,150,170,190,210,230,250]
    max_batch_size = 50
    pad_token = '\n\n\n' 
    num_pad_toks_between_two_facts = 1
    record_responses_for_debug = False
    
    save_dir = os.path.join('./artifacts/ar_experiment', f'{start_datetime_str}_{model_name.split("/")[-1]}_seeds_{"_".join([str(x) for x in seeds])}')
    os.makedirs(save_dir, exist_ok=True)
    res_per_num_facts = {}
    for num_of_facts in num_of_facts_to_test:
        res_per_num_facts[num_of_facts] = []

    for seed in seeds:
        ar_df = pd.read_csv('./ar_data.csv', dtype = str)
        ar_df = ar_df.sample(frac=1,random_state=seed) # shuffle the data to get a random AR sequence for each seed
        
        start_fact_idx = 0
        end_fact_idx = 0
        for num_of_facts in num_of_facts_to_test: # number of 'facts' in the context

            # Generate Shared Context
            start_fact_idx = end_fact_idx
            end_fact_idx = end_fact_idx + num_of_facts

            cur_df = ar_df.iloc[start_fact_idx:end_fact_idx]
            context = []
            for i_fact, fact in cur_df.iterrows():
                context += tokenizer.encode(fact["key"]) + tokenizer.encode(' ') + tokenizer.encode(fact["value"]) + num_pad_toks_between_two_facts * tokenizer.encode(pad_token)
            queries = [tokenizer.encode(fact_key) + tokenizer.encode(' ')  for fact_key in cur_df['key'].iloc[0:num_of_facts].tolist()]
            query_len_toks = [len(query) for query in queries]
            query_len_padded = max(query_len_toks)
            queries = [tokenizer.encode(pad_token) * (query_len_padded - query_len_toks[i_query]) + queries[i_query] for i_query in range(len(queries))] # padding the query lets us batch
            answers = [fact_value for fact_value in cur_df['value'].iloc[0:num_of_facts].tolist()]

            
            # Evaluate over all possible queries
            print(f'############### num of facts: {num_of_facts} ################')
            scores = []

            # queries_to_record = [10] # 30 facts
            queries_to_record = [x for x in range(num_of_facts)] # all facts
            input_ids = []
            for i_query in range(num_of_facts):
                if i_query not in queries_to_record:
                    continue
                prompt = context + tokenizer.encode(pad_token) + queries[i_query]
                input_ids.append(torch.tensor(prompt, dtype=torch.int32))
                
            input_ids = torch.vstack(input_ids).to(config['device'])
            responses = []
            num_iters = int(np.ceil(input_ids.shape[0] / max_batch_size))
            for i in tqdm(range(num_iters)):
                start_idx = i * max_batch_size
                end_idx = min((i+1) * max_batch_size, input_ids.shape[0])
                outputs = mamba.generate(input_ids[start_idx:end_idx], max_length=input_ids.shape[1]+30)
                responses.extend(tokenizer.batch_decode(outputs[:, input_ids.shape[1]:]))

            for i_resp, resp in enumerate(responses):
                parsed_response = resp.split(pad_token)[0]
                # parsed_response = find_five_digit_number(response)
                scores.append(parsed_response==answers[i_resp])

            print(f'seed: {seed}')
            print(f'number of facts: {num_of_facts}')
            print(f'number of hits: {np.array(scores).sum()}')
            print(f'accuracy: {np.array(scores).sum() / num_of_facts:.2f}')
            print(f'context length: {input_ids.shape[1]}')

            # save results 
            res_per_num_facts[num_of_facts].append(int(np.array(scores).sum()))
            results_file_name = f'final_results.json'
            save_path_results = os.path.join(save_dir, results_file_name)
            with open(save_path_results, 'w') as f:       
                json.dump(res_per_num_facts, f) 

            if record_responses_for_debug: 
                cur_df = cur_df.iloc[queries_to_record]
                cur_df['scores'] = scores
                cur_df.num_of_tokens = input_ids.shape[1]
                file_name = f'num_facts_{num_of_facts}.csv'
                save_path = os.path.join(save_dir, file_name)
                cur_df.to_csv(save_path, index=False)
    
    print(f'Final results over {len(seeds)} seeds - num hits:')
    print(tabulate([['num hits:'] + [f'{np.mean(v):.2f}' for v in res_per_num_facts.values()]], headers=['num facts:'] + [f'{k}' for k in res_per_num_facts.keys()] , tablefmt='pretty'))
    print(f'\nFinal results over {len(seeds)} seeds - accuracy:')
    print(tabulate([['accuracy:'] + [f'{np.mean(v)/k:.2f}' for k,v in res_per_num_facts.items()]], headers=['num facts:'] + [f'{k}' for k in res_per_num_facts.keys()] , tablefmt='pretty'))


if __name__ == '__main__':
    
    pd.options.mode.chained_assignment = None  # default='warn'
    config = {
        'device': 'cuda',
        'cache_dir': '/data/sls/scratch/assafbk/hf_home',
        'model_name': 'state-spaces/mamba2-370m'
        }
    start_datetime_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    run_ar_experiment(config, start_datetime_str)