%%writefile a.py
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import json
import os 
import gc
import time # For timing

import torch
import sklearn # For version check

# --- For Multiprocessing ---
from multiprocessing import Pool, cpu_count
from functools import partial # To pass fixed arguments to pool.map

# --- Config Class (Keep as is) ---
class Config:
    def __init__(self, **kwargs):
        # Attention Core
        self.hidden_size: int = 512
        self.num_attention_heads: int = 8
        self.attention_dropout: float = 0.1
        self.attention_bias: bool = False 

        # RoPE related
        self.qk_rope_head_dim: int = 64 
        self.qk_nope_head_dim: int = 0  
        self.v_head_dim: int = self.hidden_size // self.num_attention_heads 
        
        self.max_position_embeddings: int = 2048 
        self.rope_theta: float = 10000.0
        self.rope_scaling_factor: float = 1.0 

        # LoRA-like projections in ARCAttention
        self.q_lora_rank: int = self.hidden_size 
        self.kv_lora_rank: int = self.hidden_size 

        # Norms
        self.rms_norm_eps: float = 1e-6

        # Pattern Attention Modules (MPA)
        self.num_self_pattern_modules: int = 0 
        self.num_cross_pattern_modules: int = 0 
        self.pattern_heads: int = 4 

        # MLP / FFN
        self.hidden_act: str = "silu" 
        self.intermediate_size: int = int(self.hidden_size * 8 / 3) 
        
        # MoE specific
        self.n_routed_experts: int = 0 
        self.moe_intermediate_size: int = self.hidden_size // 2 
        self.num_experts_per_tok: int = 1 
        self.ep_rank: int = 0 
        self.experts_per_rank: int = self.n_routed_experts 

        self._attn_implementation: str = "normal" 

        self.pad: int = 0
        self.bos: int = 1
        self.eos: int = 2
        self.train_ctx: int = 10
        self.ex_start: int = 11
        self.input_grid: int = 12
        self.output_grid: int = 13
        self.ex_end: int = 14
        self.test_ctx: int = 15

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Config trying to set unknown attribute {key}")
        
        self.__post_init__()

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        
        calculated_v_head_dim = self.hidden_size // self.num_attention_heads
        if self.v_head_dim != calculated_v_head_dim:
            print(f"Warning: Config v_head_dim ({self.v_head_dim}) differs from calculated "
                  f"hidden_size/num_attention_heads ({calculated_v_head_dim}). Using configured v_head_dim.")

        total_qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        if self.num_attention_heads * total_qk_head_dim != self.hidden_size and total_qk_head_dim > 0:
             pass 
        elif total_qk_head_dim == 0 and self.num_attention_heads > 0:
            print("Warning: qk_rope_head_dim and qk_nope_head_dim are both 0. Defaulting qk_nope_head_dim.")
            self.qk_nope_head_dim = self.hidden_size // self.num_attention_heads

        if self.n_routed_experts > 0 and self.num_experts_per_tok > self.n_routed_experts:
            print(f"Warning: num_experts_per_tok ({self.num_experts_per_tok}) > n_routed_experts ({self.n_routed_experts}). "
                  f"Setting num_experts_per_tok to {self.n_routed_experts}.")
            self.num_experts_per_tok = self.n_routed_experts

# Global config object - this can be problematic with multiprocessing if not handled carefully.
# For functions run in separate processes, they might need their own config instantiation or have it passed.
# However, if Config is simple and only read, it might be okay (pickled).
_config_instance = Config() 

# --- End of Config Class ---

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression # Option for simpler model

def clear_cache():
    gc.collect()

# MODIFIED: Simplified MLPRegressor parameters
def get_model(model_cache=None, n_samples_for_fit=0, use_mlp=True): # Added use_mlp flag
    if not use_mlp:
        return LinearRegression() # Simpler, faster model

    use_early_stopping = True
    if n_samples_for_fit < 10: 
        use_early_stopping = False
    
    # Faster MLP settings
    max_iters = 150 # Reduced from 700
    n_iter_no_change_val = 10 # Reduced
    hidden_size = (100,) # Reduced from (100,)

    # For very few samples, even simpler MLP
    if n_samples_for_fit <= 5:
        hidden_size = (5,)
        max_iters = 50
        n_iter_no_change_val = 5


    if model_cache is None:
        if use_early_stopping:
            model_cache = MLPRegressor(hidden_layer_sizes=hidden_size, 
                                       max_iter=max_iters, 
                                       random_state=42,
                                       early_stopping=True, 
                                      # n_iter_no_change=n_iter_no_change_val,
                                      # learning_rate_init=0.01, # Can help converge faster
                                      ) # Slightly more regularization
        else:
            model_cache = MLPRegressor(hidden_layer_sizes=hidden_size, 
                                       max_iter=max_iters, 
                                       random_state=42,
                                       #early_stopping=False,
                                       #learning_rate_init=0.01,
                                      # alpha=0.001
                                      )
    return model_cache

def create_dim_pred(loaded_data): # loaded_data is task_meta
    train_inputs_for_shape_model, train_outputs_for_shape_model = [], []

    for example in loaded_data['train']:
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        train_inputs_for_shape_model.append(input_grid.shape[:2])
        train_outputs_for_shape_model.append(output_grid.shape[:2])

    for test_case in loaded_data.get('test', []):
        if 'output' in test_case:
            input_grid = np.array(test_case['input'])
            output_grid = np.array(test_case['output'])
            train_inputs_for_shape_model.append(input_grid.shape[:2])
            train_outputs_for_shape_model.append(output_grid.shape[:2])

    if not train_inputs_for_shape_model:
        last_unsolved_test_input_grid = None
        for test_case in reversed(loaded_data.get('test', [])):
            if 'output' not in test_case:
                last_unsolved_test_input_grid = np.array(test_case['input'])
                return last_unsolved_test_input_grid, last_unsolved_test_input_grid.shape[:2] # Input shape as fallback
        return None, (0,0) # Fallback if no unsolved and no train examples

    num_shape_train_samples = len(train_inputs_for_shape_model)
    
    # Decide whether to use MLP or a simpler LinearRegression based on sample size
    # If very few samples, LinearRegression might be more robust and faster.
    use_mlp_predictor = True
    if num_shape_train_samples < 3 : # Heuristic: if less than 3 examples, MLP is likely to struggle
        use_mlp_predictor = False

    shape_model = get_model(n_samples_for_fit=num_shape_train_samples, use_mlp=use_mlp_predictor)
    
    X_fit = np.array(train_inputs_for_shape_model)
    y_fit = np.array(train_outputs_for_shape_model)
    if X_fit.ndim == 1: X_fit = X_fit.reshape(-1, 1)
    if y_fit.ndim == 1: y_fit = y_fit.reshape(-1, 1) # Should generally be (N,2) already

    try:
        shape_model.fit(X_fit, y_fit)
    except ValueError as e: # Handles cases like "Expected 2D array, got 1D array instead" if shapes are weird
        # print(f"Warning: Shape model fitting error for task {loaded_data.get('id', 'Unknown')}: {e}. Defaulting shape.")
        # Fallback: predict output shape as same as input for the last test case
        last_unsolved_test_input_grid = None
        for test_case in reversed(loaded_data.get('test', [])):
            if 'output' not in test_case:
                last_unsolved_test_input_grid = np.array(test_case['input'])
                return last_unsolved_test_input_grid, last_unsolved_test_input_grid.shape[:2]
        return None, (0,0) # Further fallback

    last_unsolved_test_input_grid = None
    predicted_output_shape_tuple = None

    # Find the last test case without an 'output' to predict for
    target_test_case_input_grid = None
    for test_case in reversed(loaded_data.get('test', [])):
        if 'output' not in test_case:
            target_test_case_input_grid = np.array(test_case['input'])
            last_unsolved_test_input_grid = target_test_case_input_grid # Keep track of the grid itself
            break
    
    if target_test_case_input_grid is not None:
        predicted_shape_array = shape_model.predict([target_test_case_input_grid.shape[:2]])
        predicted_output_shape_tuple = tuple(np.maximum(1, np.round(predicted_shape_array[0])).astype(int))
    elif loaded_data.get('test'): # All test cases have output, predict for the last one
        target_test_case_input_grid = np.array(loaded_data['test'][-1]['input'])
        # last_unsolved_test_input_grid remains None as it's not "unsolved"
        predicted_shape_array = shape_model.predict([target_test_case_input_grid.shape[:2]])
        predicted_output_shape_tuple = tuple(np.maximum(1, np.round(predicted_shape_array[0])).astype(int))
    else: # No test cases at all
        # This case means create_arrays might not have a test input to build prompt with.
        # The calling function (create_arrays) should handle this.
        # For create_dim_pred, we return a default shape.
        predicted_output_shape_tuple = (0,0) # Default shape

    # `last_unsolved_test_input_grid` is returned for consistency, even if prediction was for a solved one.
    # `create_arrays` primarily uses the `predicted_output_shape_tuple`.
    # The returned grid is mostly for debugging or if `create_arrays` needed it.
    
    # If last_unsolved_test_input_grid is still None but we got a shape, it means we predicted for a solved one
    # or there were no test cases. If create_arrays needs *an* input grid that matches the prediction context,
    # we could return `target_test_case_input_grid` instead.
    # Current create_arrays doesn't directly use the returned grid, only the shape.
    
    clear_cache() # gc.collect()
    return last_unsolved_test_input_grid, predicted_output_shape_tuple


# This function will be called by each worker in the multiprocessing pool
# It needs all its arguments passed directly or be able to access them globally (carefully)
def process_single_task_for_multiprocessing(task_meta, all_solutions_data_global, max_seq_len_global, C_global):
    """
    Wrapper for create_arrays to be used with multiprocessing.Pool.
    Handles passing of global-like data that child processes need.
    """
    task_id = task_meta['id']
    # We pass C_global directly to create_arrays_modified if it accepts it,
    # or create_arrays_modified uses its own global _config_instance.
    # For simplicity, let's assume create_arrays_modified can take C as an argument.
    return create_arrays_modified(
        task_json_data=task_meta,
        true_lable_data=all_solutions_data_global,
        max_seq_len=max_seq_len_global,
        task_id=task_id,
        C_config=C_global # Pass the config object
    )

# MODIFIED create_arrays to accept C_config
def create_arrays_modified(task_json_data, true_lable_data, max_seq_len, task_id, C_config):
    # C_config is the config object passed in
    C = C_config

    if task_id is not None and task_id in true_lable_data:
        try:
            output_grid_2d = true_lable_data[task_id][0]['output']
            test_output_actual_flat = [cell for row in output_grid_2d for cell in row]
        except (KeyError, IndexError, TypeError):
            test_output_actual_flat = [C.pad] # Use pad token for missing target
    else:
        test_output_actual_flat = [C.pad] 

    target_seq = torch.tensor(test_output_actual_flat, dtype=torch.long)
    target_seq = target_seq[:max_seq_len]
    target_seq = torch.nn.functional.pad(target_seq, (0, max_seq_len - target_seq.shape[0]), "constant", C.pad)

    prompt = [C.bos, C.train_ctx] 
    for ex in task_json_data.get("train", []):
        try:
            flat_input = [item for sublist in ex["input"] for item in sublist]
            flat_output = [item for sublist in ex["output"] for item in sublist]
            prompt += [C.ex_start, C.input_grid, *flat_input, C.output_grid, *flat_output, C.ex_end]
        except (IndexError, TypeError, ValueError): 
             continue 

    prompt.append(C.test_ctx)

    if task_json_data.get("test") and task_json_data["test"][0].get("input") is not None:
        try:
            test_input_grid = task_json_data["test"][0]["input"]
            flat_test_input = [item for sublist in test_input_grid for item in sublist]
            prompt += [C.ex_start, C.input_grid] 
            prompt += flat_test_input + [C.output_grid]
        except (IndexError, KeyError, TypeError, ValueError):
            prompt += [C.output_grid] # If test input malformed, still add output grid token
    else: 
        prompt += [C.ex_start, C.input_grid, C.output_grid] # No test input, but provide context


    prompt.append(C.eos)

    # Pad or truncate prompt
    current_len = len(prompt)
    if current_len > max_seq_len:
        final_prompt = prompt[:max_seq_len-1] + [C.eos] # Ensure EOS if truncated
        is_trunc = True
        pad_len = 0
    else:
        final_prompt = prompt
        is_trunc = False
        pad_len = max_seq_len - current_len
        final_prompt.extend([C.pad] * pad_len)
        
    pad_mask = [False] * min(current_len, max_seq_len) + [True] * max(0, pad_len if not is_trunc else max_seq_len - (current_len if current_len < max_seq_len else max_seq_len) )
    pad_mask = pad_mask[:max_seq_len]


    prompt_tensor = torch.tensor(final_prompt, dtype=torch.long).unsqueeze(0)
    prompt_pad_mask_tensor = torch.tensor(pad_mask, dtype=torch.bool).unsqueeze(0)
    

    # Ensure positive length
    expected_test_output_len_flat = 0


    return {
        "task_id": task_id, 
        "prompt_tensor": prompt_tensor,
        "prompt_pad_mask_tensor": prompt_pad_mask_tensor,
      #  "expected_test_output_shape_predicted": predicted_test_output_shape_tuple, 
        "expected_test_output_len_flat_predicted": expected_test_output_len_flat, 
        "target_seq": target_seq,
      #  "is_trunc": is_trunc
    }


# --- create_patches, _dataset, create_dataset (Keep as is from previous good answer) ---
def create_patches(train_list_data, solution_list_data, split='train', limited=0):
    all_task_dataset_list = []
    for task_id, task_content in train_list_data.items(): # tqdm here is very fast, no need to parallelize
        if task_id in solution_list_data: 
            all_task_dataset_list.append({
                "id": task_id,
                "train": task_content["train"],
                "test": task_content["test"]
            })
    if limited > 0 :
      all_task_dataset_list = all_task_dataset_list[:limited]
    print(f"Prepared task list with {len(all_task_dataset_list)} tasks (after limit and solution check).")
    return all_task_dataset_list
    
def _dataset(all_task_dataset_list, ratio=0.1, split='train'):
    if not all_task_dataset_list:
        return [], []
    if len(all_task_dataset_list) == 1:
        if ratio > 0 and ratio < 1:
             print("Warning: Only 1 task available, cannot split. Using for training.")
        return all_task_dataset_list, []

    actual_test_size = ratio
    if int(len(all_task_dataset_list) * ratio) == 0 and ratio > 0 and len(all_task_dataset_list) > 1:
        actual_test_size = 1 
    elif int(len(all_task_dataset_list) * (1-ratio)) == 0 and ratio < 1 and len(all_task_dataset_list) > 1:
        actual_test_size = len(all_task_dataset_list) - 1

    train_tasks_list, val_tasks_list = train_test_split(
        all_task_dataset_list,
        test_size=actual_test_size,
        random_state=42 
    )
    return train_tasks_list, val_tasks_list

def create_dataset(challenges_data, solutions_data, split='train', limited=0, ratio=0.1):
    data = create_patches(challenges_data, solutions_data, split, limited)
    return _dataset(data, ratio=ratio)


# --- main function MODIFIED for multiprocessing ---
def main():
    start_time = time.time()
    print(f"Using scikit-learn version: {sklearn.__version__}")

    parser = argparse.ArgumentParser(description="Dataset Preparation with Multiprocessing")
    default_train_path = "/kaggle/input/arc-prize-2024/arc-agi_training_challenges.json"
    if not os.path.exists(default_train_path) and os.path.exists("arc-agi_training_challenges.json"):
        default_train_path = "arc-agi_training_challenges.json"
    
    default_solution_path = "/kaggle/input/arc-prize-2024/arc-agi_training_solutions.json"
    if not os.path.exists(default_solution_path) and os.path.exists("arc-agi_training_solutions.json"):
        default_solution_path = "arc-agi_training_solutions.json"

    # For Kaggle, usually use /kaggle/input/arc...
    # Update these if your dataset has a different path structure
    parser.add_argument('--train_path', type=str, default="/kaggle/input/arc2025-dataset/train.json", help="Path to train data json file")
    parser.add_argument('--solution_path', type=str, default="/kaggle/input/arc2025-dataset/solution.json", help="Path to solution data json file")

    parser.add_argument('--max_seq_len', type=int, default=_config_instance.max_position_embeddings, help="Max sequence length")
    parser.add_argument('--dataset_limit', type=int, default=0, help="Limit tasks (0 for no limit)")
    parser.add_argument('--validation_ratio', type=float, default=0.1, help="Validation split ratio")
    parser.add_argument('--num_workers', type=int, default=max(1, cpu_count() -1), help="Number of worker processes")


    args = parser.parse_args()
    print(f"Running with args: {args}")


    try:
        with open(args.train_path, 'r') as f:
            all_challenges_data = json.load(f)
        print(f"Loaded {len(all_challenges_data)} tasks from {args.train_path}")
    except FileNotFoundError:
        print(f"Error: File not found {args.train_path}")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: JSON decode error {args.train_path}")
        return None, None

    try:
        with open(args.solution_path, 'r') as f:
            all_solutions_data = json.load(f)
        print(f"Loaded {len(all_solutions_data)} solutions from {args.solution_path}")
    except FileNotFoundError:
        print(f"Error: File not found {args.solution_path}")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: JSON decode error {args.solution_path}")
        return None, None

    # This config instance will be passed to workers.
    # Ensure Config class is defined at the top level of the script.
    current_config_instance = Config()


    train_task_metas, val_task_metas = create_dataset(
        all_challenges_data, 
        all_solutions_data, 
        limited=args.dataset_limit,
        ratio=args.validation_ratio
    )
    print(f"Split into {len(train_task_metas)} training tasks and {len(val_task_metas)} validation tasks.")

    # Prepare for multiprocessing
    # These variables will be "global" to the worker function's scope due to `partial`
    # Or, if they are large, consider mechanisms like shared memory if performance becomes an issue
    # For dicts like all_solutions_data, pickling by multiprocessing usually handles it.
    
    # Use `partial` to fix arguments for the worker function
    # The first argument to pool.map (or imap_unordered) is the function,
    # the second is an iterable of the arguments that change per call (task_meta).
    
    worker_func_train = partial(process_single_task_for_multiprocessing, 
                                all_solutions_data_global=all_solutions_data, 
                                max_seq_len_global=args.max_seq_len,
                                C_global=current_config_instance) # Pass the config
    
    worker_func_val = partial(process_single_task_for_multiprocessing, 
                              all_solutions_data_global=all_solutions_data, 
                              max_seq_len_global=args.max_seq_len,
                              C_global=current_config_instance) # Pass the config


    processed_train_data = []
    processed_val_data = []

    # Using multiprocessing.Pool
    # tqdm can be integrated with Pool.imap_unordered for progress bars
    if args.num_workers > 0 and len(train_task_metas) > 0:
        print(f"\nProcessing {len(train_task_metas)} training tasks with {args.num_workers} workers...")
        with Pool(processes=args.num_workers) as pool:
            # Using imap_unordered can be slightly faster as results are yielded as they complete
            # tqdm integration:
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


    end_time = time.time()
    print(f"\nSuccessfully processed {len(processed_train_data)} training samples and {len(processed_val_data)} validation samples.")
    print(f"Total processing time: {end_time - start_time:.2f} seconds.")

    if processed_train_data:
        print("\n--- Example: First Processed Training Sample (task_id, shapes) ---")
        s = processed_train_data[0]
        print(f"Task ID: {s['task_id']}, Prompt Tensor Shape: {s['prompt_tensor'].shape}, Target Shape: {s['target_seq'].shape}")
    if processed_val_data:
        print("\n--- Example: First Processed Validation Sample (task_id, shapes) ---")
        s = processed_val_data[0]
        print(f"Task ID: {s['task_id']}, Prompt Tensor Shape: {s['prompt_tensor'].shape}, Target Shape: {s['target_seq'].shape}, ")


    return processed_train_data, processed_val_data

if __name__ == "__main__":
    # Ensure dummy files are not created if real ones are expected by default paths
    # (Dummy file creation logic removed for clarity, as main now handles path defaults)
    
    # The _config_instance is created when the script is imported/run.
    # For multiprocessing, it's better to pass config explicitly or ensure it's simple enough to be pickled.
    # Here, we create current_config_instance in main and pass it.

    processed_train, processed_val = main()
    print(processed_train[0])
    if processed_train is not None and processed_val is not None:
        print(f"Main function finished. Got {len(processed_train)} train and {len(processed_val)} val items.")
    else:
        print("Main function may have encountered an error or returned None.")