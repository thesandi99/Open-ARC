import torch
from config import config
from utils.mlpregressor import create_dim_pred 

def create_arrays(task_json_data, true_lable_data, max_seq_len, task_id, C_config):
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
    

    return {
        "task_id": task_id, 
        "prompt_tensor": prompt_tensor,
        "prompt_pad_mask_tensor": prompt_pad_mask_tensor,
        "target_seq": target_seq,
        "is_trunc": is_trunc
    }

def prepare_1d(task_json_data, trueth_map, task_id,
                            max_src_len=2048, max_tgt_len=1024):
    # This function was not using create_dim_pred, so I'm leaving it as is
    # unless you want to integrate shape prediction here too.
    C = config
    current_src_seq = [C.bos]
    num_train_examples = len(task_json_data['train'])

    for i, example in enumerate(task_json_data['train']):
        current_src_seq.extend(example['input'][0])
        current_src_seq.append(C.eos)
        current_src_seq.extend(example['output'][0])
        if i < num_train_examples - 1:
            current_src_seq.append(C.ex_start)

    current_src_seq.append(C.ex_start)
    current_src_seq.extend(task_json_data['test'][0]['input'][0])
    current_src_seq.append(C.eos)

    if task_id in trueth_map:
        test_output_actual = trueth_map[task_id][0]
    else:
        raise ValueError(f"Truth not found for task_id {task_id}")

    current_tgt_seq_full = [C.eos] + test_output_actual + [C.pad if hasattr(C, 'pad') else 0]

    src_len_actual = len(current_src_seq)
    src_padding_needed = max_src_len - src_len_actual
    src_padding_mask_list = [False] * src_len_actual + [True] * src_padding_needed if src_padding_needed > 0 else [False] * max_src_len
    final_src_seq = current_src_seq[:max_src_len] + ([C.pad if hasattr(C, 'pad') else 0] * max(0, src_padding_needed))
    final_src_seq = final_src_seq[:max_src_len]
    src_padding_mask_list = src_padding_mask_list[:max_src_len]

    tgt_cat_input_list = current_tgt_seq_full[:-1]
    tgt_cat_label_list = current_tgt_seq_full[1:]

    tgt_input_len_actual = len(tgt_cat_input_list)
    tgt_input_padding_needed = (max_tgt_len -1) - tgt_input_len_actual
    final_tgt_cat_input_seq = tgt_cat_input_list[:max_tgt_len-1] + ([C.pad if hasattr(C, 'pad') else 0] * max(0, tgt_input_padding_needed))
    final_tgt_cat_input_seq = final_tgt_cat_input_seq[:max_tgt_len-1]
    tgt_input_padding_mask_list = [False] * tgt_input_len_actual + [True] * tgt_input_padding_needed if tgt_input_padding_needed > 0 else [False] * (max_tgt_len-1)
    tgt_input_padding_mask_list = tgt_input_padding_mask_list[:max_tgt_len-1]

    tgt_label_len_actual = len(tgt_cat_label_list)
    tgt_label_padding_needed = (max_tgt_len -1) - tgt_label_len_actual
    final_tgt_cat_label_seq = tgt_cat_label_list[:max_tgt_len-1] + ([C.pad if hasattr(C, 'pad') else 0] * max(0, tgt_label_padding_needed))
    final_tgt_cat_label_seq = final_tgt_cat_label_seq[:max_tgt_len-1]

    src_tensor = torch.tensor(final_src_seq, dtype=torch.long).unsqueeze(0)
    tgt_cat_input_tensor = torch.tensor(final_tgt_cat_input_seq, dtype=torch.long).unsqueeze(0)
    tgt_cat_label_tensor = torch.tensor(final_tgt_cat_label_seq, dtype=torch.long).unsqueeze(0)
    src_pad_mask_tensor = torch.tensor(src_padding_mask_list, dtype=torch.bool).unsqueeze(0)
    tgt_input_pad_mask_tensor = torch.tensor(tgt_input_padding_mask_list, dtype=torch.bool).unsqueeze(0)

    return {
        "src_cat": src_tensor,
        "tgt_cat_input": tgt_cat_input_tensor,
        "tgt_cat_label": tgt_cat_label_tensor,
        "src_padding_mask": src_pad_mask_tensor,
        "tgt_padding_mask": tgt_input_pad_mask_tensor,
    }