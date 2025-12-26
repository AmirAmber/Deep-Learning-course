import os
import json
import matplotlib.pyplot as plt
import numpy as np
import glob

def plot_all_results():
    base_dir = './artifacts/ar_experiment'

    # Find all 'final_results.json' files in the subdirectories
    result_files = glob.glob(os.path.join(base_dir, '*', 'final_results.json'))

    if not result_files:
        print("No result files found! Make sure the experiment has finished at least one model.")
        return

    plt.figure(figsize=(12, 8))

    # Colors for distinction - match user's graph
    color_map = {
        'mamba2-370m': 'blue',
        'mamba2-780m': 'green',
        'mamba2-1.3b': 'orange',
        'mamba2-2.7b': 'red',
        'gpt-neo': 'grey',
        'transformer': 'grey',
        'transformer++': 'grey'
    }
    fallback_colors = ['cyan', 'magenta', 'yellow', 'black']

    print(f"Found {len(result_files)} experiments. Filtering for latest runs...")

    # Dictionary to store the latest file for each unique model
    # Key: standardized model name, Value: (timestamp, file_path)
    latest_runs = {}

    for file_path in result_files:
        folder_name = os.path.basename(os.path.dirname(file_path))
        
        # Extract timestamp and model name
        # Folder format: YYYY_MM_DD__HH_MM_SS_modelname
        try:
            # The timestamp is the first 20 chars: "YYYY_MM_DD__HH_MM_SS"
            timestamp_str = folder_name[:20]
            model_part = folder_name[21:]
            
            # Remove seeds suffix if present
            if '_seeds' in model_part:
                model_part = model_part.split('_seeds')[0]
            
            # Standardize model name for grouping
            name_lower = model_part.lower()

            # Skip pythia as requested
            if 'pythia' in name_lower:
                print(f"[SKIP] {folder_name}: Contains 'pythia'")
                continue
                
            # STRICT FILTER: 
            # 1. Ban GPT-Neo - ALLOWED NOW
            # if 'gpt-neo' in name_lower:
            #     print(f"[SKIP] {folder_name}: Contains 'gpt-neo' (Banned)")
            #     continue
            # 2. Ban generic "transformer" if it's not the new "transformerpp"
            if 'transformer' in name_lower and 'pp' not in name_lower:
                print(f"[SKIP] {folder_name}: Generic 'transformer' without 'pp' (Banned)")
                continue

            standard_name = model_part # Default
            
            # Map to canonical names
            known_names = {
                'gpt-neo-2.7b': 'Transformer-2.7B',
                'transformerpp-2.7b': 'Transformer++ 2.7B',
                'mamba2-370m': 'Mamba2-370M',
                'mamba2-780m': 'Mamba2-780M',
                'mamba2-1.3b': 'Mamba2-1.3B',
                'mamba2-2.7b': 'Mamba2-2.7B'
            }
            
            for k, v in known_names.items():
                if k in name_lower:
                    standard_name = v
                    break
            
            print(f"Checking folder: {folder_name}")
            
            # Simple string comparison for timestamps works because format is YYYY_MM_DD...
            if standard_name not in latest_runs:
                latest_runs[standard_name] = (timestamp_str, file_path)
                print(f"  -> Added as new latest for {standard_name}")
            else:
                current_latest_ts = latest_runs[standard_name][0]
                if timestamp_str > current_latest_ts:
                    latest_runs[standard_name] = (timestamp_str, file_path)
                    print(f"  -> Updated latest for {standard_name}")
                else:
                    print(f"  -> Old run, skipping (current latest: {current_latest_ts})")
                    
        except Exception as e:
            print(f"Skipping malformed folder name: {folder_name} ({e})")
            continue

    # Check for missing transformerpp results and warn the user
    if 'Transformer++ 2.7B' not in latest_runs:
        print("\n" + "!"*60)
        print("WARNING: Transformer++ results (state-spaces/transformerpp-2.7b) NOT found.")
        print("The graph cannot plot it because the experiment hasn't finished (or started) for this model yet.")
        print("Please run the experiment script: python ar_experiment_proj.py")
        print("!"*60 + "\n")

    print(f"Plotting {len(latest_runs)} unique models: {list(latest_runs.keys())}")

    for idx, (model_name, (_, file_path)) in enumerate(latest_runs.items()):
        
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Sort by number of facts (keys are strings in JSON, need int)
        x_values = sorted([int(k) for k in data.keys() if len(data[str(k)]) > 0], key=lambda x: int(x))
        if not x_values:
            print(f"Skipping {model_name} - no data found")
            continue
            
        # Calculate accuracy (sum / k)
        y_values = [np.mean(data[str(k)])/k for k in x_values]

        # Determine color
        # distinct color for each model based on map
        c_key = [k for k in color_map.keys() if k in model_name.lower()]
        if c_key:
            color = color_map[c_key[0]]
        else:
            color = fallback_colors[idx % len(fallback_colors)]
            
        plt.plot(x_values, y_values, marker='o', linestyle='-', label=model_name, color=color, linewidth=2)

    plt.title('Associative Recall: Mamba vs Transformer Scaling', fontsize=16)
    plt.xlabel('Key-Value Pairs To Remember', fontsize=14)
    plt.ylabel('Associative Recall Accuracy', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xticks(x_values)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)

    output_file = 'final_comparison_graph.png'
    plt.savefig(output_file, dpi=300)
    print(f"\nGraph saved successfully to: {output_file}")

if __name__ == "__main__":
    plot_all_results()