import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Tuple # For type hinting

from openarc.model.loss import OpenARCLoss

# Helper function to create the combined mask for decoder-style attention
def create_combined_attention_mask(
    padding_mask_bool: torch.Tensor, # (bsz, seq_len), True where padded
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Creates a combined causal and padding attention mask.
    The resulting mask is additive (-inf for masked, 0 for not masked).
    Shape: (bsz, seq_len, seq_len) for ARCAttention.
    """
    bsz = padding_mask_bool.shape[0]

    # Causal mask (additive)
    # For query position i, key positions j > i are masked.
    causal_mask = torch.full((seq_len, seq_len), 0.0, device=device, dtype=dtype)
    
    # Mask upper triangle (j > i)
    mask_values_to_fill = torch.finfo(dtype).min if dtype.is_floating_point else torch.iinfo(dtype).min
    
    # Create a boolean mask for positions to fill with -inf
    bool_causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    causal_mask.masked_fill_(bool_causal_mask, mask_values_to_fill)
    causal_mask = causal_mask.unsqueeze(0)  # Shape: (1, seq_len, seq_len) for broadcasting with bsz

    key_padding_mask_additive = torch.zeros(bsz, seq_len, device=device, dtype=dtype)
    key_padding_mask_additive.masked_fill_(padding_mask_bool, mask_values_to_fill)
    
    # Expand to (bsz, 1, seq_len) so it broadcasts correctly when added to (bsz, seq_len, seq_len) or (1, seq_len, seq_len)
    key_padding_mask_additive = key_padding_mask_additive.unsqueeze(1)
    combined_mask = causal_mask + key_padding_mask_additive # Broadcasting makes it (bsz, seq_len, seq_len)
    return combined_mask


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler], # For Automatic Mixed Precision
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    config, # Model/global config object (e.g., containing C.pad)
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    gradient_clip_val: Optional[float] = None
):
    model.train()
    total_loss = 0.0
    loss_fct = nn.CrossEntropyLoss(ignore_index=config.pad)

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} Training")

    for batch_idx, batch_data in progress_bar:
        # Assuming batch_data is a dict from a collate_fn like:
        # {'prompt_tensor': (bsz, seq_len), 'prompt_pad_mask_tensor': (bsz, seq_len)}
        input_ids_full = batch_data['prompt_tensor'].to(device) # (bsz, full_seq_len)
        
        # For next-token prediction: input is sequence up to T-1, labels are sequence from 1 to T
        current_input_ids = input_ids_full[:, :-1] # (bsz, actual_seq_len_for_input - 1)
        labels = input_ids_full[:, 1:].clone()     # (bsz, actual_seq_len_for_input - 1)
        
        # Padding mask corresponds to `input_ids_full`. We need it for `current_input_ids`.
        padding_mask_bool_for_inputs = batch_data['prompt_pad_mask_tensor'][:, :-1].to(device)

        bsz, current_seq_len = current_input_ids.shape
        
        # Create attention mask for `current_input_ids`
        # Model's dtype can be inferred from a parameter
        model_dtype = next(model.parameters()).dtype
        attention_mask = create_combined_attention_mask(
            padding_mask_bool_for_inputs,
            current_seq_len,
            device,
            dtype=model_dtype
        )
        
        optimizer.zero_grad()

        if scaler:  # Use AMP
            with torch.cuda.amp.autocast():
                # Model forward expects input_ids and attention_mask.
                # Past_key_values and initial_jmodule_states are None for standard training from scratch.
                # use_cache should be False.
                model_outputs = model(
                    input_ids=current_input_ids,
                    attention_mask=attention_mask,
                    use_cache=False
                )
                # Assuming OpenARC model returns (logits, ...) or just logits
                logits = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
                
                # logits shape: (bsz, current_seq_len, vocab_size)
                # labels shape: (bsz, current_seq_len)
                loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        
            scaler.scale(loss).backward()
            if gradient_clip_val:
                scaler.unscale_(optimizer) # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
        else:  # No AMP
            model_outputs = model(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                use_cache=False
            )
            logits = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
            loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
            loss.backward()
            if gradient_clip_val:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_val)
            optimizer.step()

        if scheduler:
            # Common to step scheduler per batch for some types (e.g., cosine decay)
            scheduler.step() 

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{total_loss/(batch_idx+1):.4f}")

    avg_epoch_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} - Average Training Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss


@torch.no_grad() # Decorator for disabling gradient calculations
def eval_one_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int, # Or could be just a string like "Validation"
    config # Model/global config object
):
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    loss_fct = nn.CrossEntropyLoss(ignore_index=config.pad)

    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch} Validation")

    for batch_idx, batch_data in progress_bar:
        input_ids_full = batch_data['prompt_tensor'].to(device)
        current_input_ids = input_ids_full[:, :-1]
        labels = input_ids_full[:, 1:].clone()
        
        padding_mask_bool_for_inputs = batch_data['prompt_pad_mask_tensor'][:, :-1].to(device)
        bsz, current_seq_len = current_input_ids.shape
        
        model_dtype = next(model.parameters()).dtype
        attention_mask = create_combined_attention_mask(
            padding_mask_bool_for_inputs,
            current_seq_len,
            device,
            dtype=model_dtype
        )

        # No AMP context needed for evaluation typically, but ensure dtype consistency if model was trained with AMP
        model_outputs = model(
            input_ids=current_input_ids,
            attention_mask=attention_mask,
            use_cache=False # Not using KV cache for full sequence validation loss
        )
        logits = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
            
        loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{total_loss/(batch_idx+1):.4f}")

    avg_epoch_loss = total_loss / len(val_loader)
    print(f"Epoch {epoch} - Average Validation Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    config_to_save: Optional[Dict[str, Any]] = None,
    scaler_state_dict: Optional[Dict] = None # Added
):
    """Saves model checkpoint."""
    save_obj = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if config_to_save:
        save_obj['config_params'] = config_to_save
    
    if scaler_state_dict: # Added
        save_obj['scaler_state_dict'] = scaler_state_dict
    
    try:
        torch.save(save_obj, filepath)
        print(f"Model checkpoint saved to {filepath}")
    except Exception as e:
        print(f"Error saving model checkpoint: {e}")