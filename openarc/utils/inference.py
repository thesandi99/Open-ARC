import torch
import json 

from typing import Optional, List, Dict, Any

from openarc.model.model import OpenARC
from openarc.config.config import config
from openarc.utils.engine import create_combined_attention_mask

import torch.nn.functional as F

def top_k_filtering(logits, top_k):
    """Masks logits for sampling to keep only top_k tokens."""
    if top_k is None or top_k == 0 or top_k >= logits.shape[-1]: 
        return logits
    
    values, _ = torch.topk(logits, top_k)
    min_values = values[:, -1].unsqueeze(-1)
    
    # Ensure mask value is compatible with logits dtype
    mask_value = torch.finfo(logits.dtype).min if logits.is_floating_point() else torch.iinfo(logits.dtype).min
    return torch.where(logits < min_values, torch.full_like(logits, mask_value), logits)

def create_inference_prompt(
    task_json_data: Dict[str, Any],
    C_config=config,
    task_id: Optional[str] = "inference_task" # Add default task_id
) -> torch.Tensor:
    """
    Creates the initial prompt sequence tensor for inference, without padding or truncation.
    """
    C = C_config

    prompt = [C.bos, C.train_ctx]
    for ex in task_json_data.get("train", []):
        try:
            flat_input = [item for sublist in ex["input"] for item in sublist]
            flat_output = [item for sublist in ex["output"] for item in sublist]
            prompt += [C.ex_start, C.input_grid, *flat_input, C.output_grid, *flat_output, C.ex_end]
        except (IndexError, TypeError, ValueError, KeyError): # Added KeyError
            # Skip malformed examples silently during inference prompt creation
            continue

    prompt.append(C.test_ctx)

    # Use the first test input if available
    test_input_found = False
    if task_json_data.get("test"):
        try:
            test_input_grid = task_json_data["test"][0]["input"]
            flat_test_input = [item for sublist in test_input_grid for item in sublist]
            prompt += [C.ex_start, C.input_grid]
            prompt += flat_test_input + [C.output_grid]
            test_input_found = True
        except (IndexError, KeyError, TypeError, ValueError):
            # If test input malformed or missing, fall back
            pass

    if not test_input_found:
        # Default if no valid test input found
        prompt += [C.ex_start, C.input_grid, C.output_grid]

    prompt_tensor = torch.tensor(prompt, dtype=torch.long).unsqueeze(0) # Add batch dimension

    return prompt_tensor


@torch.no_grad()
def generate_sequence(
    model: OpenARC,
    prompt_ids: torch.Tensor, # Shape: (1, prompt_len)
    device: torch.device,
    config=config,
    max_new_tokens: int = 100,
    temperature: float = 0.7, # Use a default temperature
    top_k: Optional[int] = 50 # Use a default top_k
) -> List[int]:
    """
    Generates a sequence autoregressively using the OpenARC model.
    Includes debugging prints.
    """
    model.eval()
    bsz, prompt_len = prompt_ids.shape
    if bsz != 1:
        raise ValueError("Batch size must be 1 for this inference function.")

    print(f"Debug: Vocab size from config: {config.output_size}")
    print(f"Debug: Using top_k={top_k}, temperature={temperature}")

    current_ids = prompt_ids.to(device)
    generated_ids = list(prompt_ids.squeeze().tolist())

    past_key_values_attn = None
    jmodule_states = None
    model_dtype = next(model.parameters()).dtype

    # --- Step 0: Process the initial prompt ---
    print("Debug: Processing initial prompt...")
    prompt_padding_mask = torch.zeros(1, prompt_len, dtype=torch.bool, device=device)
    try:
        initial_attention_mask = create_combined_attention_mask(
            padding_mask_bool=prompt_padding_mask,
            seq_len=prompt_len,
            device=device,
            dtype=model_dtype
        )
    except Exception as e:
        print(f"Error creating initial attention mask: {e}")
        raise

    try:
        outputs = model(
            input_ids=current_ids,
            attention_mask=initial_attention_mask,
            past_key_values_attn=None,
            initial_jmodule_states=None,
            use_cache=True
        )
        logits = outputs[0]
        past_key_values_attn = outputs[1]
        jmodule_states = outputs[2]
    except Exception as e:
        print(f"Error during initial model forward pass: {e}")
        raise

    # Check initial logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("Warning: Initial logits contain NaN or Inf values.")
        # Decide how to handle: raise error, return prompt, etc.
        raise ValueError("Initial logits are invalid (NaN/Inf).")

    next_token_logits = logits[:, -1, :]
    print(f"Debug: Initial next_token_logits stats: min={next_token_logits.min().item():.4f}, max={next_token_logits.max().item():.4f}, mean={next_token_logits.mean().item():.4f}")


    # --- Generation Loop ---
    print("Debug: Starting generation loop...")
    for step in range(max_new_tokens):
        # Apply sampling strategy
        scaled_logits = next_token_logits / temperature
        filtered_logits = top_k_filtering(scaled_logits, top_k)

        # --- Debugging Checks ---
        if torch.isnan(scaled_logits).any() or torch.isinf(scaled_logits).any():
            print(f"Warning: Step {step}: scaled_logits contain NaN/Inf.")
            raise ValueError(f"Step {step}: scaled_logits invalid.")
        if torch.isnan(filtered_logits).any() or torch.isinf(filtered_logits).any():
             # Check if all filtered logits are -inf
            if (filtered_logits == (torch.finfo(filtered_logits.dtype).min if filtered_logits.is_floating_point() else torch.iinfo(filtered_logits.dtype).min)).all():
                 print(f"Error: Step {step}: All filtered_logits are -inf after top_k filtering. TopK might be too restrictive or logits are degenerate.")
                 # You could try sampling from scaled_logits instead, or just raise
                 # probs = F.softmax(scaled_logits, dim=-1) # Option: fallback
                 raise ValueError(f"Step {step}: filtered_logits invalid (all -inf).")
            else:
                 print(f"Warning: Step {step}: filtered_logits contain NaN/Inf.")
                 raise ValueError(f"Step {step}: filtered_logits invalid.")
        # --- End Debugging Checks ---

        probs = F.softmax(filtered_logits, dim=-1)

        # --- More Debugging ---
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print(f"Error: Step {step}: probs contain NaN/Inf after softmax.")
            print(f"  Filtered logits stats before softmax: min={filtered_logits.min().item():.4f}, max={filtered_logits.max().item():.4f}")
            raise ValueError(f"Step {step}: probs invalid.")
        if not torch.isclose(probs.sum(dim=-1), torch.tensor(1.0, device=device, dtype=probs.dtype), atol=1e-3):
             print(f"Warning: Step {step}: probs do not sum to 1 (sum={probs.sum(dim=-1).item():.4f}).")
             # Don't necessarily raise here, but it's a red flag. Could be floating point inaccuracies if close.

        print(f"Debug: Step {step}: probs stats: min={probs.min().item():.4e}, max={probs.max().item():.4e}, sum={probs.sum().item():.4f}")
        # --- End More Debugging ---

        try:
            next_token_id = torch.multinomial(probs, num_samples=1)
        except RuntimeError as e:
            print(f"Error: Step {step}: torch.multinomial failed: {e}")
            print(f"  Probabilities causing failure (first 50): {probs.squeeze()[:50].tolist()}")
            # Re-raise the error after printing details
            raise e

        token_id_item = next_token_id.item()
        generated_ids.append(token_id_item)
        print(f"Debug: Step {step}: Generated token ID: {token_id_item}")


        if token_id_item == config.eos:
            print(f"Debug: Step {step}: EOS token generated. Stopping.")
            break

        # Prepare inputs for the next step
        current_ids = next_token_id
        
        # Minimal mask should be sufficient when using cache
        step_attention_mask = torch.zeros(1, 1, device=device, dtype=model_dtype)

        try:
            outputs = model(
                input_ids=current_ids,
                attention_mask=step_attention_mask,
                past_key_values_attn=past_key_values_attn,
                initial_jmodule_states=jmodule_states,
                use_cache=True
            )
            logits = outputs[0]
            past_key_values_attn = outputs[1]
            jmodule_states = outputs[2]
        except Exception as e:
             print(f"Error: Step {step}: Model forward pass failed: {e}")
             raise

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Warning: Step {step}: Logits from model contain NaN/Inf.")
            raise ValueError(f"Step {step}: Model produced invalid logits.")

        next_token_logits = logits[:, -1, :]
        print(f"Debug: Step {step}: Next token logits stats: min={next_token_logits.min().item():.4f}, max={next_token_logits.max().item():.4f}, mean={next_token_logits.mean().item():.4f}")

    print("Debug: Generation loop finished or max_new_tokens reached.")
    return generated_ids 

def load_state(checkpoint):
    print(f"Loading model architecture...")
    model = OpenARC()
    device = "cuda:0"
    print(f"Loading model weights from: {checkpoint}")
    try:
        # Load state dict (ensure map_location for CPU compatibility if needed)
        checkpoint = torch.load(checkpoint, map_location=device)
        # Adjust loading based on how checkpoint was saved (might be nested)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint) # Assume raw state dict
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {checkpoint}")
        exit()
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        exit()

    model.to(device)
    model.eval()
    return model


def predict_outcome(model_path, json_path, task_id_for_infrence=None, max_new_tokens=2048, temp=0.7, top_k=20, device="cuda"):
    try:
        # Set weights_only=True for security unless you absolutely trust the source
        with open(json_path, 'r') as f: all_challenges_data = json.load(f)
    except Exception as e:
        print(f"Error creating inference prompt: {e}")
    
    if not all_challenges_data:
        print("Error: No tasks found in the challenge data.")
        exit(1)
    
    if task_id_for_infrence is None:
        task_id, task_obj = next(iter(all_challenges_data.items()))
    
    try:
        prompt_tensor = create_inference_prompt(task_obj, task_id=task_id)
        print(f"Prompt length: {prompt_tensor.shape[1]} tokens")
    except Exception as e:
        print(f"Error creating inference prompt: {e}")
        exit(1)

    model = load_state(model_path)
    print(f"Generating sequence (max_new_tokens={max_new_tokens}, temp={temp}, top_k={top_k})...")
    generated_token_ids = None # Initialize before try block
    try:
        generated_token_ids = generate_sequence(
            model=model,
            prompt_ids=prompt_tensor,
            device=device,
            config=config,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_k=top_k
        )
        print("Generation complete.")
    except Exception as e:
        print(f"Error during sequence generation: {e}")
    print(generated_token_ids)
    return generated_token_ids
