import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import json
import os
import time
from tqdm import tqdm

from multiprocessing import cpu_count 
import gc 

import sys
sys.path.append('/kaggle/working/Open-ARC')

from openarc.model.model import OpenARC 
from openarc.config.config import config as current_config  
from openarc.utils.engine import train_one_epoch, eval_one_epoch, save_model_checkpoint
from openarc.data.dataset import _create_data

class ARCDataset(Dataset):
    def __init__(self, processed_samples_list):
        self.samples = [s for s in processed_samples_list if s is not None]
        if not self.samples:
            print("Warning: ARCDataset initialized with no valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Ensure tensors are squeezed if they come with an extra batch dim from create_arrays
        return {
            "task_id": sample["task_id"],
            "prompt_tensor": sample["prompt_tensor"].squeeze(0) if sample["prompt_tensor"].ndim == 2 else sample["prompt_tensor"],
            "prompt_pad_mask_tensor": sample["prompt_pad_mask_tensor"].squeeze(0) if sample["prompt_pad_mask_tensor"].ndim == 2 else sample["prompt_pad_mask_tensor"],
            "target_seq": sample["target_seq"].squeeze(0) if sample["target_seq"].ndim == 2 else sample["target_seq"]
        }


def main():
    
    # current_config is now the global instance from openarc.config.config
    parser = argparse.ArgumentParser(description="OpenARC Model Training")
    
    # Paths for ARC data
    parser.add_argument('--train_challenges_path', type=str, default="arc-agi_training_challenges.json", help="Path to training challenges JSON file")
    parser.add_argument('--train_solutions_path', type=str, default="arc-agi_training_solutions.json", help="Path to training solutions JSON file")
    default_max_seq_len = getattr(current_config, 'max_position_embeddings', 2048)
    parser.add_argument('--max_seq_len', type=int, default=default_max_seq_len, help="Max sequence length for model and data processing")
    parser.add_argument('--dataset_limit', type=int, default=0, help="Limit number of tasks processed (0 for no limit)")
    parser.add_argument('--validation_ratio', type=float, default=0.1, help="Ratio of data to use for validation")
    parser.add_argument('--num_workers', type=int, default=max(0, cpu_count() - 2), help="Number of workers for data preprocessing in dataset.py")

    # Training loop parameters
    parser.add_argument('--batch_size', type=int, default=getattr(current_config, 'batch_size', 16))
    parser.add_argument('--epochs', type=int, default=getattr(current_config, 'num_epochs', 5))
    parser.add_argument('--lr', type=float, default=getattr(current_config, 'learning_rate', 5e-5))
    parser.add_argument('--weight_decay', type=float, default=getattr(current_config, 'weight_decay', 0.01))
    parser.add_argument('--gradient_clip_val', type=float, default=getattr(current_config, 'gradient_clip_val', 1.0))
    parser.add_argument('--use_amp', action='store_true', default=getattr(current_config, 'use_amp', True)) # Allow config default
    parser.add_argument('--output_dir', type=str, default="openarc_checkpoints")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Seed 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    
    current_config.max_position_embeddings = args.max_seq_len
    current_config.batch_size = args.batch_size 
    current_config.learning_rate = args.lr   
    current_config.use_amp = args.use_amp    


    print("Effective Configuration for this run:")
    print(f"  Global Config (e.g., current_config.max_position_embeddings): {current_config.max_position_embeddings}")
    print(f"  Command-line args (used for training loop and data processing calls):")
    for k, v in vars(args).items(): print(f"    {k}: {v}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading raw data...")
    try:
        with open(args.train_challenges_path, 'r') as f: all_challenges_data = json.load(f)
        with open(args.train_solutions_path, 'r') as f: all_solutions_data = json.load(f)
    except FileNotFoundError as e: print(f"Error: {e}. Ensure paths are correct. Exiting."); return
    except json.JSONDecodeError as e: print(f"Error decoding JSON: {e}. Check file integrity. Exiting."); return
    print(f"Loaded {len(all_challenges_data)} challenges and {len(all_solutions_data)} solutions.")

    start_data_time = time.time()
    train_samples_list, val_samples_list = _create_data(
        train_data_raw=all_challenges_data,
        solutions_data_raw=all_solutions_data,
        limit=args.dataset_limit, 
        args=args                 
    )
    
    del all_challenges_data, all_solutions_data; gc.collect() # Free memory
    print(f"Data processing via dataset.py finished in {time.time() - start_data_time:.2f}s.")

    if not train_samples_list: print("No training samples processed. Exiting."); return

    train_dataset = ARCDataset(train_samples_list)
    val_dataset = ARCDataset(val_samples_list) if val_samples_list else None
    del train_samples_list, val_samples_list; gc.collect()


    dl_num_workers = min(args.num_workers if args.num_workers > 0 else 0, 4)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=dl_num_workers, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=dl_num_workers, pin_memory=True if device.type == 'cuda' else False) if val_dataset and len(val_dataset) > 0 else None
    print(f"Created DataLoaders. Train batches: {len(train_loader)}, Val batches: {len(val_loader) if val_loader else 0}")

    if len(train_loader) == 0: print("Train loader is empty. Exiting."); return
    print("Initializing model...")
    
    # Model uses current_config (the global, updated instance)
    model = OpenARC(config=current_config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {total_params:,} trainable parameters.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Use args.use_amp as it reflects the direct CLI choice for this run.
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == 'cuda' else None
    if scaler: print("Using Automatic Mixed Precision (AMP).")
    
    scheduler = None # Placeholder for LR scheduler if needed

    start_epoch = 0
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}...")
        try:
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            if scaler and 'scaler_state_dict' in checkpoint: scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"Resumed from epoch {start_epoch -1}. Next epoch: {start_epoch}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint properly: {e}. Starting training from scratch.")
            start_epoch = 0

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        train_loss = train_one_epoch(
            model, optimizer, scaler, train_loader, device, epoch,
            config=current_config, scheduler=scheduler, gradient_clip_val=args.gradient_clip_val
        )

        val_loss_str = "N/A"
        current_loss_for_saving = train_loss
        if val_loader:
            val_loss = eval_one_epoch(model, val_loader, device, epoch, config=current_config)
            val_loss_str = f"{val_loss:.4f}"
            current_loss_for_saving = val_loss
        
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss_str}, Time: {time.time() - epoch_start_time:.2f}s")

        checkpoint_name_default = f"epoch_{epoch}_loss_{current_loss_for_saving:.4f}.pt"
        final_checkpoint_name = checkpoint_name_default

        if val_loader and current_loss_for_saving < best_val_loss:
            best_val_loss = current_loss_for_saving
            print(f"New best validation loss: {best_val_loss:.4f}. Saving as best_model.pt")
            final_checkpoint_name = "best_model.pt" # Save best model with a fixed name
            # save_model_checkpoint(model, optimizer, epoch, current_loss_for_saving, os.path.join(args.output_dir, checkpoint_name_default), config_to_save=vars(current_config))

        # Prepare config to save (only serializable parts from the *instance*)
        config_dict_to_save = {k: v for k, v in current_config.__dict__.items() if isinstance(v, (int, float, str, bool, list, dict, tuple))}

        save_model_checkpoint(
            model=model, optimizer=optimizer, epoch=epoch, loss=current_loss_for_saving,
            filepath=os.path.join(args.output_dir, final_checkpoint_name),
            config_to_save=config_dict_to_save,
            scaler_state_dict=scaler.state_dict() if scaler else None # Pass scaler state
        )

    print("Training finished.")

if __name__ == "__main__":
    main()