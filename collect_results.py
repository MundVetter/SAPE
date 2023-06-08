import os
import re
import pandas as pd
import constants
import sys

def read_results_files(parent_folder):
    results_dfs = []
    unique_tags_datasets = set()
    
    for root, _, files in os.walk(parent_folder):
        for file in files:
            match = re.match(r"(.+)-results\.csv", file)
            if match:
                tag = match.groups()[0]
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                df['tag'] = tag
                results_dfs.append(df)
                
                datasets = df['dataset'].unique()
                for ds in datasets:
                    unique_tags_datasets.add((tag, ds))
    
    if not results_dfs:
        raise FileNotFoundError(f"No '*-results.csv' files were found in '{parent_folder}'.")
    
    return pd.concat(results_dfs), unique_tags_datasets

def calculate_mean_std(df, column):
    return df[column].mean(), df[column].std()

def main(parent_folder):
    results_df, unique_tags_datasets = read_results_files(parent_folder)

    # Sort based on the tag
    unique_tags_datasets = sorted(unique_tags_datasets, key=lambda x: x[0])

    for tag, ds in unique_tags_datasets:
        print(f"Tag: {tag}, Dataset: {ds}")
        print("Mean, Standard Deviation")
        
        specific_results_df = results_df[(results_df['tag'] == tag) & (results_df['dataset'] == ds)]
        mean, std = calculate_mean_std(specific_results_df, 'result')
        print(f"{mean}, {std}")
        print("\n")

if __name__ == "__main__":
    folder = "2d_images"

    # check argv[1] for the folder name
    if len(sys.argv) > 1:
        folder = sys.argv[1]

    parent_folder = constants.CHECKPOINTS_ROOT / folder
    main(parent_folder)
