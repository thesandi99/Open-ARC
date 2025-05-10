import numpy as np
import argparse
import json
import os 

from openarc.data.dataset import create_dataset

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Dimension Predictor")
    parser.add_argument('--train_path', type=str, default="/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json", help="Path to train data json file")
    parser.add_argument('--solution_path', type=str, default="/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json", help="Path to solution data json file")
    
    args = parser.parse_args()
    with open(args.train_path, 'r') as f:
        train_path = json.load(f)

    with open(args.solution_path, 'r') as f:
        solution_path = json.load(f)
    
    # Handle path input
    train_dataset, val_dataset = create_dataset(train_path, solution_path, split='train', limited=0) # limited 0 means no limit 
    print("train shape:", len(train_dataset), "val Shape:", len(val_dataset))
    print(train_dataset[0])
    return None

if __name__ == "__main__":
    main()