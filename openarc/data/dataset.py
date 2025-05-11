from tqdm import tqdm
from sklearn.model_selection import train_test_split

from openarc.utils.preprocess import create_arrays 
from openarc.config.config import Config as GlobalConfigClass 
from openarc.config.config import config as C_global_config_instance 

from multiprocessing import Pool, cpu_count
from functools import partial

from torch.utils.data import Dataset
import torch


class ARCDataset(Dataset):
    def __init__(self, processed_samples_list):
        # Filter out any None results from multiprocessing
        self.samples = [s for s in processed_samples_list if s is not None]
        if not self.samples:
            print("Warning: ARCDataset initialized with no valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "task_id": sample["task_id"],
            "prompt_tensor": sample["prompt_tensor"].squeeze(0) if sample["prompt_tensor"].ndim == 2 else sample["prompt_tensor"],
            "prompt_pad_mask_tensor": sample["prompt_pad_mask_tensor"].squeeze(0) if sample["prompt_pad_mask_tensor"].ndim == 2 else sample["prompt_pad_mask_tensor"],
            "target_seq": sample["target_seq"].squeeze(0) if sample["target_seq"].ndim == 2 else sample["target_seq"]
        }
    
def create_patches(train_list_data, solution_list_data, limited=0):
    all_task_dataset_list = []
    for task_id, task_content in tqdm(train_list_data.items(), desc="Creating task patches"):
        if task_id in solution_list_data:
            all_task_dataset_list.append({
                "id": task_id,
                "train": task_content.get("train", []),
                "test": task_content.get("test", [])
            })
    
    return all_task_dataset_list if limited == 0 else all_task_dataset_list[:limited]
    
def _split_dataset(all_task_dataset_list, ratio=0.1):
    if not all_task_dataset_list:
        return [], []
    if len(all_task_dataset_list) == 1 and 0 < ratio < 1:
        print("Warning: Only 1 task available for split. Using for training, validation empty.")
        return all_task_dataset_list, []
    if ratio == 0.0:
        return all_task_dataset_list, []
    if ratio == 1.0:
        return [], all_task_dataset_list
        
    train_tasks_list, val_tasks_list = train_test_split(
        all_task_dataset_list,
        test_size=ratio,
        random_state=42
    )
    return train_tasks_list, val_tasks_list

def _process_single_task_mp_worker(task_meta, all_solutions_data_dict, max_seq_len_val, C_config_as_dict):
    """
    Worker function for multiprocessing.
    Reconstructs a Config object from C_config_as_dict for use by create_arrays.
    """
    task_id = task_meta['id']

    # Reconstruct a Config instance for the worker process
    worker_config_instance = GlobalConfigClass() # Create a new base Config instance
    
    # Update this fresh instance with attributes from the dictionary
    for key, value in C_config_as_dict.items():
        if hasattr(worker_config_instance, key):
            setattr(worker_config_instance, key, value)

    
    worker_config_instance.max_position_embeddings = max_seq_len_val

    return create_arrays(
        task_json_data=task_meta,
        true_lable_data=all_solutions_data_dict,
        max_seq_len=max_seq_len_val, 
        task_id=task_id,
        C_config=worker_config_instance 
    )

def _create_data(train_data_raw, solutions_data_raw, limit, args):
    """
    Prepares and splits data using multiprocessing.
    `args` is the argparse object from train.py.
    `limit` here is typically `args.dataset_limit`.
    """
    
    task_patches = create_patches(
        train_list_data=train_data_raw,
        solution_list_data=solutions_data_raw,
        limited=args.dataset_limit 
    )

    train_task_metas, val_task_metas = _split_dataset(
        task_patches,
        ratio=args.validation_ratio
    )
    print(f"Split into {len(train_task_metas)} training tasks and {len(val_task_metas)} validation tasks.")

    # Use the global config INSTANCE, which might have been modified by train.py (e.g., max_position_embeddings)
    main_process_config_instance = C_global_config_instance
    
    # Create a picklable dictionary from the main process's Config instance attributes
    config_dict_for_workers = {}
    for k, v in main_process_config_instance.__dict__.items():
        # Filter for basic, picklable types. Add other types if they are known to be safe.
        if isinstance(v, (int, float, str, bool, list, dict, tuple, type(None))):
            config_dict_for_workers[k] = v
        # else:
            # print(f"Debug (main_process): Skipping unpicklable type for attribute '{k}': {type(v)}")
            
    
    config_dict_for_workers['max_position_embeddings'] = args.max_seq_len
    print(f"  [dataset.py _create_data] Using args.max_seq_len for workers: {args.max_seq_len}")

    _worker_fn_with_args = partial(
        _process_single_task_mp_worker,
        all_solutions_data_dict=solutions_data_raw,
        max_seq_len_val=args.max_seq_len, # Pass the authoritative max_seq_len
        C_config_as_dict=config_dict_for_workers # Pass the picklable dictionary
    )
    
    processed_train_data = []
    processed_val_data = []

    effective_num_workers = min(args.num_workers, cpu_count()) if args.num_workers > 0 else 0

    # Training data processing
    if train_task_metas:
        if effective_num_workers > 0:
            print(f"\nProcessing {len(train_task_metas)} training tasks with {effective_num_workers} workers...")
            with Pool(processes=effective_num_workers) as pool:
                processed_train_data = list(tqdm(pool.imap_unordered(_worker_fn_with_args, train_task_metas), total=len(train_task_metas), desc="Processing training tasks"))
        else:
            print("\nProcessing training tasks sequentially...")
            for task_meta in tqdm(train_task_metas, desc="Processing training tasks (sequential)"):
                processed_train_data.append(_worker_fn_with_args(task_meta))
    
    # Validation data processing
    if val_task_metas:
        if effective_num_workers > 0:
            print(f"\nProcessing {len(val_task_metas)} validation tasks with {effective_num_workers} workers...")
            with Pool(processes=effective_num_workers) as pool:
                processed_val_data = list(tqdm(pool.imap_unordered(_worker_fn_with_args, val_task_metas), total=len(val_task_metas), desc="Processing validation tasks"))
        else:
            print("\nProcessing validation tasks sequentially...")
            for task_meta in tqdm(val_task_metas, desc="Processing validation tasks (sequential)"):
                 processed_val_data.append(_worker_fn_with_args(task_meta))
    
    return processed_train_data, processed_val_data