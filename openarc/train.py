import numpy as np
import argparse
import json
import os 

from tqdm import tqdm

from openarc.data.dataset import _create_data

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Dimension Predictor")
    parser.add_argument('--train_path', type=str, default="/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json", help="Path to train data json file")
    parser.add_argument('--solution_path', type=str, default="/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json", help="Path to solution data json file")
    parser.add_argument('--max_seq_len', type=int, default=C.max_position_embeddings, help="Max sequence length")
    parser.add_argument('--dataset_limit', type=int, default=0, help="Limit tasks (0 for no limit)")
    parser.add_argument('--validation_ratio', type=float, default=0.1, help="Validation split ratio")
    parser.add_argument('--num_workers', type=int, default=3, help="Number of worker processes")
    args = parser.parse_args()
    with open(args.train_path, 'r') as f:
        train_path = json.load(f)

    with open(args.solution_path, 'r') as f:
        solution_path = json.load(f)
    
    # Handle path input
    train_dataset, val_dataset = _create_data(train_path, solution_path, split='train', limited=args.dataset_limit, args=args) # limited 0 means no limit


    return None

if __name__ == "__main__":
    main()