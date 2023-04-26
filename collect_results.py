import os
import pandas as pd
import numpy as np

def read_results_files(parent_folder, name, tag):
    results_dfs = []
    
    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file == f"{name}_{tag}_results.csv":
                file_path = os.path.join(root, file)
                results_dfs.append(pd.read_csv(file_path))
    
    if not results_dfs:
        raise FileNotFoundError(f"No '{name}_{tag}_results.csv' files were found in '{parent_folder}'.")
    
    return pd.concat(results_dfs)

def calculate_mean_std(df, column):
    return df[column].mean(), df[column].std()

def main(parent_folder, name, tag):
    results_df = read_results_files(parent_folder, name, tag)
    configurations = results_df['configuration'].unique()

    print("Configuration, Mean, Standard Deviation")
    for config in configurations:
        config_df = results_df[results_df['configuration'] == config]
        mean, std = calculate_mean_std(config_df, 'value')
        print(f"{config}, {mean}, {std}")

if __name__ == "__main__":
    parent_folder = "assets/checkpoints/2d_images"
    name = "test_masked"
    tag = "x"
    main(parent_folder, name, tag)
