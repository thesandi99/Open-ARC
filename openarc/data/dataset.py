from tqdm import tqdm

from sklearn.model_selection import train_test_split
from openarc.utils.preprocess import create_arrays
from openarc.config import config as C

from multiprocessing import Pool, cpu_count
from functools import partial

def create_patches(train_list, solution_list, split='train', limited=0):
    all_task_dataset_list = []
    for task_id, task_content in tqdm(train_list.items(), desc="Processing tasks"):
        if task_id in solution_list: # Only include tasks with available solutions
            all_task_dataset_list.append({
                "id": task_id,
                "train": task_content["train"],
                "test": task_content["test"]
            })
    
    return all_task_dataset_list if limited == 0 else all_task_dataset_list[:limited]
    
def _dataset(all_task_dataset_list, ratio=0.1, split='train'):
    train_tasks_list, val_tasks_list = train_test_split(
        all_task_dataset_list,
        test_size=ratio,
        random_state=42
    )
    return train_tasks_list, val_tasks_list

def create_dataset(path_train_dir, path_solution_dir, split='train', limited=0):
    data = create_patches(path_train_dir, path_solution_dir, split, limited)
    return _dataset(data)


def process_single_task_for_multiprocessing(task_meta, all_solutions_data_global, max_seq_len_global, C_global):
    """
    Wrapper for create_arrays to be used with multiprocessing.Pool.
    Handles passing of global-like data that child processes need.
    """
    task_id = task_meta['id']
    return create_arrays(
        task_json_data=task_meta,
        true_lable_data=all_solutions_data_global,
        max_seq_len=max_seq_len_global,
        task_id=task_id,
        C_config=C_global
    )


def _create_data(train_data, solutions_data, limit, args):
    train_task_metas, val_task_metas = create_dataset(
        train_data, 
        solutions_data, 
        limited=args.dataset_limit,
        ratio=args.validation_ratio
    )
    print(f"Split into {len(train_task_metas)} training tasks and {len(val_task_metas)} validation tasks.")

    # Prepare for multiprocessing
    worker_func_train = partial(process_single_task_for_multiprocessing, 
                                all_solutions_data_global=solutions_data, 
                                max_seq_len_global=args.max_seq_len,
                                C_global=C) 
    
    worker_func_val = partial(process_single_task_for_multiprocessing, 
                              all_solutions_data_global=solutions_data, 
                              max_seq_len_global=args.max_seq_len,
                              C_global=C) 
    
    processed_train_data = []
    processed_val_data = []

    # Using multiprocessing.Pool
    if args.num_workers > 0 and len(train_task_metas) > 0:
        print(f"\nProcessing {len(train_task_metas)} training tasks with {args.num_workers} workers...")
        with Pool(processes=args.num_workers) as pool:
            processed_train_data = list(tqdm(pool.imap_unordered(worker_func_train, train_task_metas), total=len(train_task_metas), desc="Processing training tasks"))
    
    elif len(train_task_metas) > 0: # Single worker fallback (or num_workers=0)
        print("\nProcessing training tasks sequentially...")
        for task_meta in tqdm(train_task_metas, desc="Processing training tasks"):
            processed_train_data.append(worker_func_train(task_meta)) # Call directly


    if args.num_workers > 0 and len(val_task_metas) > 0:
        print(f"\nProcessing {len(val_task_metas)} validation tasks with {args.num_workers} workers...")
        with Pool(processes=args.num_workers) as pool:
            processed_val_data = list(tqdm(pool.imap_unordered(worker_func_val, val_task_metas), total=len(val_task_metas), desc="Processing validation tasks"))
    
    elif len(val_task_metas) > 0: # Single worker fallback
        print("\nProcessing validation tasks sequentially...")
        for task_meta in tqdm(val_task_metas, desc="Processing validation tasks"):
             processed_val_data.append(worker_func_val(task_meta))
    
    return processed_train_data, processed_val_data