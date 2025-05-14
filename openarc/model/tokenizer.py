# tokenization
# 10(bos) 13(train)  11([ start) 0 0 ...  12(] end) 11 0 0 1 01 .... 12  14(test) 16(pred-ex)11 [ 11[..]12 12 11[.. 12]..12] .. 16(pred-ex) 17(eos)

"""
# Example Usage:

import json
from openarc.model import OpenTokenizer

config = Config()

task_json_data = '{"train":[{"input":[[0,0,4,0,0,0,0,0,4,4,4,4,4,4,4,4,4,4,0,0,0,0,4,0,0,0,0,0,0,0,0,0]],"output":[[0,0,0,0,0,0,0,0,4,4,4,4,4,4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]},{"input":[[0,0,0,2,0,0,2,0,0,2,0,0,2,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,0,0]],"output":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,0,0]]},{"input":[[4,4,4,4,4,4,4,4,4,4,4,4,4,4,0,0,0,4,0,0,0,0,4,0,0,0,0,0,0,0,0,0]],"output":[[4,4,4,4,4,4,4,4,4,4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]}],"test":[{"input":[[3,3,3,3,3,3,3,3,3,3,3,3,0,0,3,0,0,3,0,0,0,3,0,0,0,0,0,0,0,0,0,0]]}]}'
solution = '{"1d_denoising_1c_0":[[3,3,3,3,3,3,3,3,3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]}'
solution_ = json.loads(solution)
task_json_data = json.loads(task_json_data)

tokenizer = OpenTokenizer.from_pretrained(config_instance=config)
prompt = tokenizer.tokenize_task(task_json_data=task_json_data, true_label_data=solution_, task_id="1d_denoising_1c_0", max_seq_len=8192)
print(prompt)

"""

import torch
from openarc.config.config import Config

"""
example tokenization:
[[10, 13, 18, 11,  0,  0,  4,  0,  0,  0,  0,  0,  4,  4,  4,  4,  4,  4,
          4,  4,  4,  4,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         12, 19, 11,  0,  0,  0,  0,  0,  0,  0,  0,  4,  4,  4,  4,  4,  4,  4,
          4,  4,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 12,
         18, 11,  0,  0,  0,  2,  0,  0,  2,  0,  0,  2,  0,  0,  2,  0,  2,  2,
          2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0, 12, 19,
         11,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  2,  2,
          2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0, 12, 18, 11,
          4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  0,  0,  0,  4,
          0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0, 12, 19, 11,  4,
          4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 12, 14, 16, 18, 11,
          3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  0,  0,  3,  0,  0,  3,
          0,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 12]]

Label Target Tensor: tensor([[19, 11,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 12, 17]])
"""


class OpenTokenizer:
    def __init__(self, config_instance: Config):
        self.config = config_instance
        self.model_name = ["OpenARC-FlatGrid"] 

    def check_name(self): return self.model_name

    @classmethod
    def from_pretrained(cls, config_instance: Config = None):
        if config_instance is None: 
            config_instance = Config() 
        return cls(config_instance=config_instance)

    @property
    def vocab_size_prop(self):
        return self.config.vocab_size 

          
    def tokenize_task(self, task_json_data, true_label_data, task_id, max_seq_len):
        C_config = self.config
        
        # Prompt Sequence (X) 
        prompt_list = [C_config.bos, C_config.train_ctx]
        
        # Training examples
        for ex in task_json_data.get("train", []):
            try:
                if not (isinstance(ex.get("input"), list) and all(isinstance(r, list) for r in ex["input"]) and \
                        isinstance(ex.get("output"), list) and all(isinstance(r, list) for r in ex["output"])):
                    continue 
                
                flat_input = [item for sublist in ex["input"] for item in sublist if isinstance(item, int)]
                flat_output = [item for sublist in ex["output"] for item in sublist if isinstance(item, int)]
                
                prompt_list.append(C_config.input_grid_token)
                prompt_list.append(C_config.grid_start_token)
                prompt_list.extend(flat_input)
                prompt_list.append(C_config.grid_end_token)
                
                prompt_list.append(C_config.output_grid_token)
                prompt_list.append(C_config.grid_start_token)
                prompt_list.extend(flat_output)
                prompt_list.append(C_config.grid_end_token)
            except:
                continue 

        prompt_list.append(C_config.test_ctx)
        
        prompt_ends_with_output_grid_marker = False 
        processed_first_test_input_for_prompt = False

        if task_json_data.get("test") and isinstance(task_json_data["test"], list) and task_json_data["test"]:
            test_example_data = task_json_data["test"][0]

            flat_test_input = []
            has_valid_test_input_structure = False
            if isinstance(test_example_data, dict) and isinstance(test_example_data.get("input"), list) and \
               all(isinstance(r, list) for r in test_example_data["input"]):
                has_valid_test_input_structure = True

            if has_valid_test_input_structure:
                try:
                    flat_test_input = [item for sublist in test_example_data["input"] for item in sublist if isinstance(item, int)]
                    
                    if C_config.use_pred_ex_format:
                        flat_test_output_context = []
                        has_valid_test_output_in_data = False 
                        if isinstance(test_example_data.get("output"), list) and \
                           all(isinstance(r, list) for r in test_example_data["output"]):
                            try:
                                flat_test_output_context = [item for sublist in test_example_data["output"] for item in sublist if isinstance(item, int)]
                                has_valid_test_output_in_data = True
                            except: pass 
                        
                        if has_valid_test_output_in_data:
                            prompt_list.extend([C_config.input_grid_token, C_config.grid_start_token])
                            prompt_list.extend(flat_test_input)
                            prompt_list.append(C_config.grid_end_token)
                            
                            prompt_list.extend([C_config.output_grid_token, C_config.grid_start_token])
                            prompt_list.extend(flat_test_output_context)
                            prompt_list.append(C_config.grid_end_token)
                            
                            prompt_list.append(C_config.pred_ex_token)
                            prompt_list.extend([C_config.input_grid_token, C_config.grid_start_token])
                            prompt_list.extend(flat_test_input)
                            prompt_list.append(C_config.grid_end_token)
                        else: 
                            prompt_list.append(C_config.pred_ex_token)
                            prompt_list.extend([C_config.input_grid_token, C_config.grid_start_token])
                            prompt_list.extend(flat_test_input)
                            prompt_list.append(C_config.grid_end_token)
                    else: 
                        prompt_list.extend([C_config.input_grid_token, C_config.grid_start_token])
                        prompt_list.extend(flat_test_input)
                        prompt_list.append(C_config.grid_end_token)
                        prompt_list.append(C_config.output_grid_token)
                        prompt_ends_with_output_grid_marker = True
                    
                    processed_first_test_input_for_prompt = True
                except:
                    processed_first_test_input_for_prompt = False 
        
        if not processed_first_test_input_for_prompt:
            if C_config.use_pred_ex_format:
                prompt_list.append(C_config.pred_ex_token)
                prompt_list.extend([C_config.input_grid_token, C_config.grid_start_token, C_config.grid_end_token])
            else:
                prompt_list.extend([C_config.input_grid_token, C_config.grid_start_token, C_config.grid_end_token])
                prompt_list.append(C_config.output_grid_token)
                prompt_ends_with_output_grid_marker = True

        is_prompt_truncated = False
        if len(prompt_list) > max_seq_len:
            final_prompt_list = prompt_list[:max_seq_len]
            is_prompt_truncated = True
        else:
            final_prompt_list = prompt_list
        
        prompt_tensor = torch.tensor(final_prompt_list, dtype=torch.long).unsqueeze(0)
        prompt_pad_mask_tensor = torch.zeros_like(prompt_tensor, dtype=torch.bool)

        # Target Sequence (Y) 
        actual_target_output_flat = [] # Default to empty
        
        if task_id is not None and true_label_data and task_id in true_label_data:
            task_solution_data = true_label_data[task_id]
            
            # Expects: "task_id": [[grid_row_1], [grid_row_2], ...]
            if isinstance(task_solution_data, list) and task_solution_data:
                # If task_solution_data is like [[r1c1, r1c2], [r2c1, r2c2]], flatten it.
                # If task_solution_data is already like [[r1c1, r1c2, r2c1, r2c2]], this will also work.
                
                try:
                    # Check if the first element is a list (indicating list of rows)
                    if isinstance(task_solution_data[0], list):
                        # Ensure all elements are lists (rows) and all cells are ints
                        if all(isinstance(row, list) for row in task_solution_data) and \
                           all(isinstance(cell, int) for row in task_solution_data for cell in row):
                            actual_target_output_flat = [cell for row in task_solution_data for cell in row]

                except TypeError: # Handles cases where an element might not be iterable (e.g. if structure is mixed)
                    actual_target_output_flat = [] # Fallback
                except: 
                    actual_target_output_flat = [] 

        target_seq_content_list = []
        if prompt_ends_with_output_grid_marker:
            target_seq_content_list.append(C_config.grid_start_token)
            target_seq_content_list.extend(actual_target_output_flat)
            target_seq_content_list.append(C_config.grid_end_token)
        else:
            target_seq_content_list.append(C_config.output_grid_token)
            target_seq_content_list.append(C_config.grid_start_token)
            target_seq_content_list.extend(actual_target_output_flat)
            target_seq_content_list.append(C_config.grid_end_token)
        
        target_seq_content_list.append(C_config.eos)
        
        if len(target_seq_content_list) > max_seq_len: 
            target_seq_list = target_seq_content_list[:max_seq_len]
            if max_seq_len > 0:
                if target_seq_list[-1] != C_config.eos:
                     target_seq_list[-1] = C_config.eos
            elif max_seq_len == 0:
                 target_seq_list = []
        else:
            target_seq_list = target_seq_content_list
        
        target_seq = torch.tensor(target_seq_list, dtype=torch.long).unsqueeze(0)

        return {
            "task_id": task_id,
            "prompt_tensor": prompt_tensor,
            "prompt_pad_mask_tensor": prompt_pad_mask_tensor,
            "target_seq_tensor": target_seq, 
            "is_prompt_truncated": is_prompt_truncated
        }

    def detokenize_grid(self, token_sequence, rows, cols):
        
        # This helper remains largely the same, it just processes a list of assumed digit tokens
        if torch.is_tensor(token_sequence):
            squeezed_tokens = token_sequence.squeeze()
            if squeezed_tokens.ndim == 0: 
                 token_sequence = [squeezed_tokens.item()] if squeezed_tokens.numel() == 1 else []
            else: 
                 token_sequence = squeezed_tokens.tolist()
        elif not isinstance(token_sequence, list):
            token_sequence = [token_sequence] if isinstance(token_sequence, (int, float)) else []

        digit_tokens = [t for t in token_sequence if isinstance(t, int) and 0 <= t <= 9]
        
        # This error message is now more specific to its context in detokenize_output_to_json
        # if not digit_tokens and token_sequence: 
        #      return [['Error: No valid digits in sequence (after stripping wrappers)']] 
        
        if not digit_tokens: # If, after stripping wrappers, no digits were found.
            if rows == 0 and cols == 0: return [[]] 
            if rows == 0 : return [] 
            # Fill with pad if shape is specified but no digits.
            return [([self.config.pad] * cols) for _ in range(rows)]


        expected_len = rows * cols
        if rows < 0 or cols < 0: return [['Error: Invalid dimensions for grid reconstruction']]
        if rows == 0: return [] 
        if cols == 0: return [[] for _ in range(rows)]

        if len(digit_tokens) < expected_len:
            digit_tokens.extend([self.config.pad] * (expected_len - len(digit_tokens)))
        else: 
            digit_tokens = digit_tokens[:expected_len]
        
        grid = []
        current_idx = 0
        for _ in range(rows):
            row_list = []
            for _ in range(cols):
                if current_idx < len(digit_tokens): 
                    row_list.append(digit_tokens[current_idx])
                    current_idx += 1
            grid.append(row_list)
        return grid

    def detokenize_output_to_json(self, predicted_token_ids, task_id_str, expected_output_shape=None):
        C_config = self.config
        
        if torch.is_tensor(predicted_token_ids):
            squeezed_ids = predicted_token_ids.squeeze()
            if squeezed_ids.ndim == 0:
                predicted_token_ids = [squeezed_ids.item()] if squeezed_ids.numel() == 1 else []
            else:
                predicted_token_ids = squeezed_ids.tolist()
        elif not isinstance(predicted_token_ids, list):
             predicted_token_ids = [predicted_token_ids] if isinstance(predicted_token_ids, (int, float)) else []

        if not predicted_token_ids:
            if expected_output_shape and isinstance(expected_output_shape, (list, tuple)) and len(expected_output_shape) == 2:
                rows, cols = expected_output_shape
                if rows >= 0 and cols >= 0:
                    return {task_id_str: self.detokenize_grid([], rows, cols)}
            return {task_id_str: [[]]}

        processed_tokens = list(predicted_token_ids) 
        if processed_tokens and processed_tokens[-1] == C_config.eos:
            processed_tokens.pop()

        if processed_tokens and processed_tokens[-1] == C_config.grid_end_token:
            processed_tokens.pop()

        grid_content_tokens = []
        if len(processed_tokens) >= 2 and \
           processed_tokens[0] == C_config.output_grid_token and \
           processed_tokens[1] == C_config.grid_start_token:
            grid_content_tokens = processed_tokens[2:]
        elif len(processed_tokens) >= 1 and \
             processed_tokens[0] == C_config.grid_start_token:
            grid_content_tokens = processed_tokens[1:]
        else: 
            grid_content_tokens = processed_tokens 

        actual_grid_cell_tokens = [tok for tok in grid_content_tokens if isinstance(tok, int) and 0 <= tok <= 9]
        
        if not actual_grid_cell_tokens and grid_content_tokens: 
            # This means there were tokens after stripping wrappers, but none were digits 0-9
            return {task_id_str: [['Error: Predicted output contained no valid digits after stripping wrappers']]}
        
        if not actual_grid_cell_tokens: # Covers empty original, or only wrappers, or only non-digits inside
            if expected_output_shape and isinstance(expected_output_shape, (list, tuple)) and len(expected_output_shape) == 2:
                rows, cols = expected_output_shape
                if rows >=0 and cols >= 0:
                     return {task_id_str: self.detokenize_grid([], rows, cols)} # Use helper for empty grid of shape
            return {task_id_str: [[]]} 

        final_grid = [[]]
        if expected_output_shape and isinstance(expected_output_shape, (list, tuple)) and len(expected_output_shape) == 2:
            rows, cols = expected_output_shape
            if rows >= 0 and cols >= 0:
                 final_grid = self.detokenize_grid(actual_grid_cell_tokens, rows, cols)
            else: 
                 final_grid = [['Error: Invalid expected_output_shape']]
        else: 
            if actual_grid_cell_tokens: 
                final_grid = [actual_grid_cell_tokens]
            else: # Should be caught above
                 final_grid = [[]]

        return {task_id_str: final_grid}
