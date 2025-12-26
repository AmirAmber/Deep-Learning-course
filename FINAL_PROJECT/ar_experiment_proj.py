import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import warnings
warnings.filterwarnings("ignore")
try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:
    try:
        from submodules.mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    except ImportError:
        MambaLMHeadModel = None
        print("MambaLMHeadModel could not be imported. Mamba models will not run.")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
import re
import gc
import traceback
import json

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

    model_name = config['model_name']
    print(f"Loading model: {model_name}")
    
    # Dynamic tokenizer loading
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b") # Fallback
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    is_mamba = 'mamba' in model_name.lower()
    
    if is_mamba:
        # NOTE: MambaLMHeadModel.from_pretrained passes kwargs to the class __init__, 
        # and some versions don't accept cache_dir in __init__. 
        # The cache_dir is used by the internal loading mechanism but we can rely on default or set it globally if needed.
        # For now, we remove it to fix the TypeError.
        model = MambaLMHeadModel.from_pretrained(model_name).to(torch.float16).to(config["device"])
    else:
        # Added trust_remote_code=True to support custom models like Transformer++
        # Added low_cpu_mem_usage=True to avoid OOM on smaller instances (requires accelerate)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=config['cache_dir'], torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True).to(config["device"])
        except (ValueError, ImportError) as e:
            if 'transformerpp' in model_name:
                print(f"AutoModel failed ({e}). Attempting fallback classes...")
                try:
                    print("--> Trying GPTNeoXForCausalLM with use_safetensors=True...")
                    model = GPTNeoXForCausalLM.from_pretrained(model_name, cache_dir=config['cache_dir'], torch_dtype=torch.float16, use_safetensors=True, low_cpu_mem_usage=True).to(config["device"])
                except Exception as e_neox:
                    print(f"GPTNeoX failed: {e_neox}")
                    print("--> Trying LlamaForCausalLM with use_safetensors=True...")
                    try:
                        from transformers import LlamaForCausalLM
                        model = LlamaForCausalLM.from_pretrained(model_name, cache_dir=config['cache_dir'], torch_dtype=torch.float16, use_safetensors=True, low_cpu_mem_usage=True).to(config["device"])
                    except Exception as e_llama:
                        print(f"Llama fallback also failed: {e_llama}")
                        raise e # Raise original error if both fail
            else:
                raise e
        model.eval()

    seeds = [123,234,345,456,567]
    num_of_facts_to_test = [10,30,50,70,90,110,130,150,170,190,210,230,250]
    
    # Dynamic batch size: Mamba is memory efficient, Transformer needs small batches on T4/L4 GPUs
    if is_mamba:
        if '2.7b' in model_name.lower():
            max_batch_size = 8 # Conservative batch size for 2.7B
        elif '1.3b' in model_name.lower():
            max_batch_size = 20
        else:
            max_batch_size = 50
    else:
        max_batch_size = 4
    
    pad_token = '\n\n\n' 
    num_pad_toks_between_two_facts = 1
    record_responses_for_debug = False
    
    # Ensure save directory uses a safe name (replace slashes)
    short_model_name = model_name.split("/")[-1]
    save_dir = os.path.join('./artifacts/ar_experiment', f'{start_datetime_str}_{short_model_name}_seeds_{"_".join([str(x) for x in seeds])}')
    os.makedirs(save_dir, exist_ok=True)
    
    res_per_num_facts = {}
    try:
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
                
                # Padding queries
                # Note: For transformers, typically left padding is better for generation, but we stick to logic here
                pad_id = tokenizer.encode(pad_token)[0] if tokenizer.encode(pad_token) else tokenizer.eos_token_id
                
                queries = [[pad_id] * (query_len_padded - query_len_toks[i_query]) + queries[i_query] for i_query in range(len(queries))] 
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
                    # For transformers, list + list works. 
                    prompt = context + tokenizer.encode(pad_token) + queries[i_query]
                    input_ids.append(torch.tensor(prompt, dtype=torch.int64)) # Use int64 for torch
                    
                input_ids = torch.vstack(input_ids) # Keep on CPU until batching
                responses = []
                num_iters = int(np.ceil(input_ids.shape[0] / max_batch_size))
                
                for i in tqdm(range(num_iters)):
                    start_idx = i * max_batch_size
                    end_idx = min((i+1) * max_batch_size, input_ids.shape[0])
                    batch_input = input_ids[start_idx:end_idx].to(config['device'])
                    
                    with torch.no_grad():
                        if is_mamba:
                            # Use max_length that accounts for input + new tokens (30 new tokens approx)
                            outputs = model.generate(batch_input, max_length=input_ids.shape[1]+30)
                            generated = outputs[:, input_ids.shape[1]:]
                        else:
                            # Transformer generation
                            outputs = model.generate(batch_input, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
                            generated = outputs[:, batch_input.shape[1]:]

                    responses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

                for i_resp, resp in enumerate(responses):
                    # Clean up response: take everything before first pad_token/newline/etc if implied
                    parsed_response = resp.strip().split(pad_token)[0].strip()
                    # Simple check
                    scores.append(parsed_response == answers[i_resp])

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

    finally:
        # Cleanup memory
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    
    pd.options.mode.chained_assignment = None  # default='warn'
    
    # List of models to run
    models_to_run = [
        # 'state-spaces/mamba2-370m',
        # 'state-spaces/mamba2-780m',
        # 'state-spaces/mamba2-1.3b',
        # 'state-spaces/mamba2-2.7b',
        # 'state-spaces/transformerpp-2.7b'
        'EleutherAI/gpt-neo-2.7B'
    ]
    
    start_datetime_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    
    for model_name in models_to_run:
        config = {
            'device': 'cuda',
            'cache_dir': './hf_cache',
            'model_name': model_name
        }
        print(f"\n\n==================================================")
        print(f"STARTING EXPERIMENT FOR: {model_name}")
        print(f"==================================================\n")
        print(f"==================================================\n")
        
        if not os.path.exists('./ar_data.csv'):
            print("ERROR: ar_data.csv not found in current directory! Please upload it.")
            continue
            
        try:
            run_ar_experiment(config, start_datetime_str)
        except Exception as e:
            error_msg = f"FAILED for {model_name}: {e}\n"
            print(error_msg)
            with open("experiment_errors.log", "a") as f:
                f.write(f"[{datetime.now()}] {error_msg}")
                f.write(traceback.format_exc() + "\n")
            traceback.print_exc()

    # Generate the comparison graph automatically
    print("\nGenerating comparison graph...")
    from plot_comparison import plot_all_results
    plot_all_results()

