import os
import re
import pandas as pd
import numpy as np
import constants

def read_results_files(parent_folder):
    results_dfs = []
    unique_names_tags_functions = set()
    
    for root, _, files in os.walk(parent_folder):
        for file in files:
            match = re.match(r"(.+)-(.+)-results\.csv", file)
            if match:
                name, tag = match.groups()
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                df['name'] = name
                df['tag'] = tag
                results_dfs.append(df)
                
                functions = df['function'].unique()
                for func in functions:
                    unique_names_tags_functions.add((name, tag, func))
    
    if not results_dfs:
        raise FileNotFoundError(f"No '*-*-*results.csv' files were found in '{parent_folder}'.")
    
    return pd.concat(results_dfs), unique_names_tags_functions

def calculate_mean_std(df, column):
    return df[column].mean(), df[column].std()

def main(parent_folder):
    results_df, unique_names_tags_functions = read_results_files(parent_folder)

    for name, tag, func in unique_names_tags_functions:
        print(f"Name: {name}, Tag: {tag}, Function: {func}")
        print("Configuration, Mean, Standard Deviation")
        
        specific_results_df = results_df[(results_df['name'] == name) & (results_df['tag'] == tag) & (results_df['function'] == func)]
        configurations = specific_results_df['configuration'].unique()
        
        for config in configurations:
            config_df = specific_results_df[specific_results_df['configuration'] == config]
            mean, std = calculate_mean_std(config_df, 'value')
            print(f"{config}, {mean}, {std}")
        
        print("\n")

if __name__ == "__main__":
    parent_folder = constants.CHECKPOINTS_ROOT / "2d_images"
    main(parent_folder)
