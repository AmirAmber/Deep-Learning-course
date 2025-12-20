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
        'gpt-neo-2.7b': 'grey',
        'gpt-neo-2.7B': 'grey',
        'transformer-2.7b': 'grey'
    }
    fallback_colors = ['cyan', 'magenta', 'yellow', 'black']

    print(f"Found {len(result_files)} experiments. Plotting...")

    for idx, file_path in enumerate(result_files):
        # Extract model name from the folder name
        folder_name = os.path.basename(os.path.dirname(file_path))
        # Folder format is: DATE_TIME_MODELNAME
        # We want just the MODELNAME part (everything after the last underscore of the date)
        # Example: 2025_12_19__14_41_02_mamba2-370m -> mamba2-370m
        try:
            # heuristic: splitting by fixed date format length
            model_name = folder_name[21:]
            # Remove seeds suffix if present
            if '_seeds' in model_name:
                model_name = model_name.split('_seeds')[0]
            
            # Prettify labels to match user image
            name_lower = model_name.lower()
            known_names = {
                'gpt-neo-2.7b': 'Transformer-2.7B',
                'mamba2-370m': 'Mamba2-370M',
                'mamba2-780m': 'Mamba2-780M',
                'mamba2-1.3b': 'Mamba2-1.3B',
                'mamba2-2.7b': 'Mamba2-2.7B'
            }
            # key matching
            for k, v in known_names.items():
                if k in name_lower:
                    model_name = v
                    break
        except:
            model_name = folder_name

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Sort by number of facts (keys are strings in JSON, need int)
        x_values = sorted([int(k) for k in data.keys()])
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