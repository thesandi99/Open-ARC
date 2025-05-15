# tokenization
# 10(bos) 13(train)  11([ start) 0 0 ...  12(] end) 11 0 0 1 01 .... 12  14(test) 16(pred-ex)11 [ 11[..]12 12 11[.. 12]..12] .. 16(pred-ex) 17(eos)

import torch
from openarc.config.config import Config

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

    def _tokenize_structured_grid(self, grid_data_list_of_lists, C_config):
        """
        Tokenizes a grid (list of lists) by adding start/end tokens
        for the overall grid AND for each row.
        Example: [[1,2],[3]] -> [11, 11, 1, 2, 12, 11, 3, 12, 12]
                 (grid_start, row1_start, 1, 2, row1_end, row2_start, 3, row2_end, grid_end)
        Handles empty grids or grids with empty rows.
        """
        if not isinstance(grid_data_list_of_lists, list):
            return [C_config.grid_start_token, C_config.grid_end_token]

        tokenized_grid = [C_config.grid_start_token]
        for row in grid_data_list_of_lists:
            if isinstance(row, list):
                tokenized_grid.append(C_config.grid_start_token)
                tokenized_grid.extend([item for item in row if isinstance(item, int) and 0 <= item <= 9])
                tokenized_grid.append(C_config.grid_end_token)
        tokenized_grid.append(C_config.grid_end_token)
        return tokenized_grid

    def tokenize_task(self, task_json_data, true_label_data, task_id, max_seq_len):
        C_config = self.config

        prompt_list = [C_config.bos, C_config.train_ctx]

        for ex in task_json_data.get("train", []):
            try:
                input_grid_data = ex.get("input")
                output_grid_data = ex.get("output")

                if not (isinstance(input_grid_data, list) and all(isinstance(r, list) for r in input_grid_data) and \
                        isinstance(output_grid_data, list) and all(isinstance(r, list) for r in output_grid_data)):
                    continue

                tokenized_input = self._tokenize_structured_grid(input_grid_data, C_config)
                tokenized_output = self._tokenize_structured_grid(output_grid_data, C_config)

                prompt_list.append(C_config.input_grid_token)
                prompt_list.extend(tokenized_input)

                prompt_list.append(C_config.output_grid_token)
                prompt_list.extend(tokenized_output)
            except Exception:
                continue

        prompt_list.append(C_config.test_ctx)
        prompt_ends_with_output_grid_marker = False
        processed_first_test_input_for_prompt = False

        if task_json_data.get("test") and isinstance(task_json_data["test"], list) and task_json_data["test"]:
            test_example_data = task_json_data["test"][0]
            raw_test_input_data = None

            if isinstance(test_example_data, dict) and isinstance(test_example_data.get("input"), list) and \
               all(isinstance(r, list) for r in test_example_data["input"]):
                raw_test_input_data = test_example_data["input"]

            if raw_test_input_data is not None:
                try:
                    tokenized_test_input = self._tokenize_structured_grid(raw_test_input_data, C_config)

                    if C_config.use_pred_ex_format:
                        raw_test_output_context_data = None
                        if isinstance(test_example_data.get("output"), list) and \
                           all(isinstance(r, list) for r in test_example_data["output"]):
                            raw_test_output_context_data = test_example_data["output"]

                        if raw_test_output_context_data is not None:
                            tokenized_test_output_context = self._tokenize_structured_grid(raw_test_output_context_data, C_config)

                            prompt_list.append(C_config.input_grid_token)
                            prompt_list.extend(tokenized_test_input)
                            prompt_list.append(C_config.output_grid_token)
                            prompt_list.extend(tokenized_test_output_context)

                            prompt_list.append(C_config.pred_ex_token)
                            prompt_list.append(C_config.input_grid_token)
                            prompt_list.extend(tokenized_test_input)
                        else:
                            prompt_list.append(C_config.pred_ex_token)
                            prompt_list.append(C_config.input_grid_token)
                            prompt_list.extend(tokenized_test_input)
                    else:
                        prompt_list.append(C_config.input_grid_token)
                        prompt_list.extend(tokenized_test_input)
                        prompt_list.append(C_config.output_grid_token)
                        prompt_ends_with_output_grid_marker = True
                    processed_first_test_input_for_prompt = True
                except Exception:
                    processed_first_test_input_for_prompt = False

        if not processed_first_test_input_for_prompt:
            empty_tokenized_grid = self._tokenize_structured_grid([], C_config)
            if C_config.use_pred_ex_format:
                prompt_list.append(C_config.pred_ex_token)
                prompt_list.append(C_config.input_grid_token)
                prompt_list.extend(empty_tokenized_grid)
            else:
                prompt_list.append(C_config.input_grid_token)
                prompt_list.extend(empty_tokenized_grid)
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

        actual_target_output_structured = []
        if task_id is not None and true_label_data and task_id in true_label_data:
            task_solution_data = true_label_data[task_id]
            if isinstance(task_solution_data, list) and \
               all(isinstance(row, list) for row in task_solution_data):
                actual_target_output_structured = task_solution_data

        tokenized_target_grid = self._tokenize_structured_grid(actual_target_output_structured, C_config)

        target_seq_content_list = []
        if prompt_ends_with_output_grid_marker:
            target_seq_content_list.extend(tokenized_target_grid)
        else:
            target_seq_content_list.append(C_config.output_grid_token)
            target_seq_content_list.extend(tokenized_target_grid)
        target_seq_content_list.append(C_config.eos)

        if len(target_seq_content_list) > max_seq_len:
            target_seq_list = target_seq_content_list[:max_seq_len]
            if max_seq_len > 0 and target_seq_list:
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

    def _detokenize_structured_grid_from_tokens(self, token_list, C_config):
        """
        Attempts to parse a list of tokens (already stripped of outer markers like 19, 17)
        back into a list of lists, assuming the nested [11 [11 row1 12] [11 row2 12] 12] structure.
        """
        grid_data = []
        if not token_list or token_list[0] != C_config.grid_start_token or token_list[-1] != C_config.grid_end_token:
            # If not wrapped in overall grid tokens, or empty, return as a single flat list (best effort)
            # This path might be taken if the model doesn't perfectly adhere to nested structure.
            return [[t for t in token_list if isinstance(t, int) and 0 <= t <= 9]]

        # We expect tokens between the outermost grid_start and grid_end
        row_tokens_blob = token_list[1:-1]
        current_row = []
        in_row = False
        for token in row_tokens_blob:
            if token == C_config.grid_start_token:
                if in_row: # Nested grid_start_token, error or treat as data? For now, reset.
                    current_row = []
                in_row = True
            elif token == C_config.grid_end_token:
                if in_row:
                    grid_data.append(current_row)
                    current_row = []
                    in_row = False
                # else: Spurious grid_end_token, ignore
            elif in_row and isinstance(token, int) and 0 <= token <= 9:
                current_row.append(token)
            # Other tokens (non-digits when in_row, or outside a row_start/row_end) are ignored.

        if in_row: # Unclosed last row
            grid_data.append(current_row)

        # If no rows were parsed but there were digits, return flat as fallback
        if not grid_data and any(isinstance(t, int) and 0 <= t <= 9 for t in row_tokens_blob):
            return [[t for t in row_tokens_blob if isinstance(t, int) and 0 <= t <= 9]]
        
        # Ensure at least [[]] if no valid rows/data found but structure was present
        if not grid_data and token_list[0] == C_config.grid_start_token and token_list[-1] == C_config.grid_end_token and len(token_list) == 2:
            return [[]]
        if not grid_data and any(t for t in row_tokens_blob if t not in [C_config.grid_start_token, C_config.grid_end_token]): # Some content but no rows
            pass # Fall through to default empty if nothing else
            
        return grid_data if grid_data else [[]] # Ensure [[]] for empty valid grid

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
            # Use expected_output_shape to return a structured empty grid if provided
            if expected_output_shape and isinstance(expected_output_shape, (list, tuple)) and len(expected_output_shape) == 2:
                rows, cols = expected_output_shape
                if rows >= 0 and cols >= 0:
                    # The old detokenize_grid produces flat padding, adjust if structured padding needed
                    # For now, this specific path with expected_output_shape might give flat padding
                    return {task_id_str: self.detokenize_grid([], rows, cols)} # Fallback to old style for this case
            return {task_id_str: [[]]} # Default empty grid

        processed_tokens = list(predicted_token_ids)
        if processed_tokens and processed_tokens[-1] == C_config.eos:
            processed_tokens.pop()

        # `grid_content_tokens_with_rows` should contain the sequence starting with the
        # overall grid_start_token and ending with the overall grid_end_token,
        # e.g., [11, 11, d, d, 12, 11, d, 12, 12]
        grid_content_tokens_with_rows = []
        if len(processed_tokens) >= 2 and \
           processed_tokens[0] == C_config.output_grid_token:
            grid_content_tokens_with_rows = processed_tokens[1:] # Remove leading 19
        else:
            grid_content_tokens_with_rows = processed_tokens # Assume it's already starting with 11 or is just content

        # Now, parse this potentially nested structure
        final_grid_list_of_lists = self._detokenize_structured_grid_from_tokens(grid_content_tokens_with_rows, C_config)

        # The `expected_output_shape` logic might need refinement if the model
        # perfectly produces the nested structure. The `_detokenize_structured_grid_from_tokens`
        # tries to reconstruct rows. If `expected_output_shape` is strict,
        # we might need to re-flatten and then reshape, or adjust row parsing.
        # For now, we trust `_detokenize_structured_grid_from_tokens`.
        # If it returns [[]] and expected_output_shape is e.g. (2,2), we might want [ [], [] ] or padded.

        if not final_grid_list_of_lists or (len(final_grid_list_of_lists) == 1 and not final_grid_list_of_lists[0]):
            # If detokenization resulted in empty or [[]]
            if expected_output_shape and isinstance(expected_output_shape, (list, tuple)) and len(expected_output_shape) == 2:
                rows, cols = expected_output_shape
                if rows > 0 and cols >= 0 : # Create empty rows structure
                     final_grid_list_of_lists = [[] for _ in range(rows)]
                     if cols == 0 : # if cols is 0, each row is empty list
                         pass
                     elif cols > 0: # if cols >0 pad each row
                         final_grid_list_of_lists = [([C_config.pad] * cols) for _ in range(rows)]
                elif rows == 0: # if rows is 0, return empty list
                    final_grid_list_of_lists = []

        return {task_id_str: final_grid_list_of_lists}

    def detokenize_grid(self, token_sequence, rows, cols): # Kept for potential fallback or other uses
        if torch.is_tensor(token_sequence):
            squeezed_tokens = token_sequence.squeeze()
            if squeezed_tokens.ndim == 0:
                 token_sequence = [squeezed_tokens.item()] if squeezed_tokens.numel() == 1 else []
            else:
                 token_sequence = squeezed_tokens.tolist()
        elif not isinstance(token_sequence, list):
            token_sequence = [token_sequence] if isinstance(token_sequence, (int, float)) else []

        digit_tokens = [t for t in token_sequence if isinstance(t, int) and 0 <= t <= 9]

        if not digit_tokens:
            if rows == 0 and cols == 0: return [[]]
            if rows == 0 : return []
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
