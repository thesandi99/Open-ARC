import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union


class OpenARCLoss(nn.Module):
    """
    Advanced Loss Function for ARC challenges combining multiple optimization objectives.
    
    This loss function uses a combination of:
    1. Cross-entropy loss (primary classification loss)
    2. Focal loss (to handle class imbalance)
    3. Label smoothing (to prevent overconfidence)
    4. Soft F1 loss (differentiable approximation of F1 score)
    5. Consistency regularization (to encourage consistent predictions)
    6. Self-distillation (knowledge distillation within the same model)
    
    All components are computed on GPU for efficiency and combined using learnable weights.
    """
    def __init__(
        self, 
        num_classes: int,
        pad_idx: int = -100,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        temperature: float = 1.0,
        consistency_weight: float = 0.1,
        distillation_weight: float = 0.1,
        learn_weights: bool = True,
        initial_weights: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Initialize the advanced ARC loss function.
        
        Args:
            num_classes: Number of classes for classification
            pad_idx: Index to ignore in loss calculation (padding)
            alpha: Weighting factor for focal loss
            gamma: Focusing parameter for focal loss
            label_smoothing: Label smoothing factor
            temperature: Temperature for knowledge distillation
            consistency_weight: Weight for consistency regularization
            distillation_weight: Weight for self-distillation
            learn_weights: Whether to learn component weights
            initial_weights: Initial weights for loss components
        """
        super(OpenARCLoss, self).__init__()
        self.num_classes = num_classes
        self.pad_idx = pad_idx
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.temperature = temperature
        self.consistency_weight = consistency_weight
        self.distillation_weight = distillation_weight
        
        # Default initial weights
        default_weights = {
            'ce_loss': 1.0,
            'focal_loss': 0.5,
            'soft_f1_loss': 0.3,
            'consistency_loss': 0.1,
            'distillation_loss': 0.1
        }
        
        # Use provided weights or defaults
        self.initial_weights = initial_weights if initial_weights is not None else default_weights
        
        # Create learnable weights if enabled
        if learn_weights:
            self.weight_params = nn.ParameterDict({
                k: nn.Parameter(torch.tensor(v, dtype=torch.float))
                for k, v in self.initial_weights.items()
            })
        else:
            self.register_buffer('weight_params', {
                k: torch.tensor(v, dtype=torch.float)
                for k, v in self.initial_weights.items()
            })
        
        # Tracking losses
        self.losses = {}
        self.metrics = {}
        
        # Cross entropy loss with label smoothing
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=pad_idx, 
            label_smoothing=label_smoothing,
            reduction='none'
        )
        
    def reset(self) -> None:
        """Reset tracked losses and metrics."""
        self.losses = {}
        self.metrics = {}
        
    def _get_weights(self) -> Dict[str, torch.Tensor]:
        """Get normalized weights for loss components."""
        # Get raw weights
        weights = {k: self.weight_params[k].abs() for k in self.weight_params}
        
        # Normalize weights to sum to 1 (softmax-like normalization)
        weight_sum = sum(weights.values())
        normalized_weights = {k: v / weight_sum for k, v in weights.items()}
        
        return normalized_weights
    
    def _compute_ce_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        return self.ce_loss(logits, labels)
    
    def _compute_focal_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss for handling class imbalance.
        
        Focal Loss: FL(p_t) = -alpha * (1-p_t)^gamma * log(p_t)
        where p_t is the model's estimated probability for the true class.
        """
        # Create one-hot encoding of labels
        one_hot = torch.zeros_like(logits)
        mask = (labels != self.pad_idx).unsqueeze(1)
        target_one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) * mask
        
        # Compute probabilities
        probs = F.softmax(logits, dim=1)
        pt = (target_one_hot * probs).sum(1)
        
        # Compute focal weights
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting for class balance
        alpha_weight = torch.ones_like(focal_weight) * self.alpha
        alpha_weight = torch.where(labels != self.pad_idx, alpha_weight, torch.zeros_like(alpha_weight))
        
        # Compute focal loss
        focal_loss = -alpha_weight * focal_weight * torch.log(pt + 1e-10)
        
        return focal_loss
    
    def _compute_soft_f1_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute differentiable soft F1 loss.
        
        This is a differentiable approximation of the F1 score used as a loss function.
        """
        # Create one-hot encoding
        one_hot = torch.zeros_like(logits)
        mask = (labels != self.pad_idx).unsqueeze(1)
        target_one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) * mask
        
        # Get predicted probabilities
        probs = F.softmax(logits, dim=1)
        
        # Compute soft TP, FP, FN
        tp = (probs * target_one_hot).sum(dim=1)
        fp = (probs * (1 - target_one_hot)).sum(dim=1)
        fn = ((1 - probs) * target_one_hot).sum(dim=1)
        
        # Compute soft F1
        soft_f1 = 2 * tp / (2 * tp + fp + fn + 1e-10)
        
        # Convert to loss (1 - F1)
        soft_f1_loss = 1 - soft_f1
        
        # Mask padded positions
        mask = (labels != self.pad_idx).float()
        soft_f1_loss = soft_f1_loss * mask
        
        return soft_f1_loss
    
    def _compute_consistency_loss(
        self, 
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency regularization loss.
        
        Encourages smooth, consistent predictions across similar examples.
        """
        # Generate slightly perturbed logits
        noise = torch.randn_like(logits) * 0.1
        perturbed_logits = logits + noise
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=1)
        perturbed_probs = F.softmax(perturbed_logits, dim=1)
        
        # KL divergence between original and perturbed
        consistency_loss = F.kl_div(
            perturbed_probs.log(),
            probs,
            reduction='none'
        ).sum(dim=1)
        
        return consistency_loss
    
    def _compute_self_distillation(
        self, 
        logits: torch.Tensor, 
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute self-distillation loss.
        
        Uses the model's own predictions as soft targets with a temperature.
        """
        # Detach original logits to create "teacher" predictions
        with torch.no_grad():
            teacher_logits = logits.detach() / temperature
            teacher_probs = F.softmax(teacher_logits, dim=1)
        
        # Student logits with temperature
        student_logits = logits / temperature
        student_log_probs = F.log_softmax(student_logits, dim=1)
        
        # KL divergence loss
        distillation_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='none'
        ).sum(dim=1) * (temperature ** 2)
        
        return distillation_loss
    
    def compute_metrics(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics efficiently on GPU where possible.
        """
        # Get predictions
        preds = logits.argmax(dim=1)
        
        # Create mask for valid labels
        valid_mask = (labels != self.pad_idx)
        
        # Compute accuracy
        correct = (preds == labels) & valid_mask
        accuracy = correct.float().sum() / valid_mask.float().sum()
        
        # Store metrics
        metrics = {'accuracy': accuracy.item()}
        
        # For other metrics that require CPU computation, use a small sample
        # to avoid excessive CPU transfers during training
        if torch.rand(1).item() < 0.05:  # Compute only 5% of the time for efficiency
            # Get CPU versions for sklearn metrics
            cpu_preds = preds[valid_mask].cpu().numpy()
            cpu_labels = labels[valid_mask].cpu().numpy()
            
            # Only compute if we have data
            if len(cpu_preds) > 0:
                try:
                    from sklearn.metrics import f1_score, cohen_kappa_score, precision_score
                    
                    # Compute F1 score
                    f1 = f1_score(
                        y_true=cpu_labels,
                        y_pred=cpu_preds,
                        average='weighted'
                    )
                    metrics['f1'] = f1
                    
                    # Compute Cohen's Kappa
                    kappa = cohen_kappa_score(
                        y1=cpu_labels,
                        y2=cpu_preds,
                        weights='quadratic'
                    )
                    metrics['kappa'] = kappa
                    
                    # Compute precision
                    precision = precision_score(
                        y_true=cpu_labels,
                        y_pred=cpu_preds,
                        average='weighted'
                    )
                    metrics['precision'] = precision
                    
                except Exception as e:
                    print(f"Error computing metrics: {e}")
        
        return metrics
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass to compute the combined loss.
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            labels: Ground truth labels (batch_size)
            return_components: Whether to return individual loss components
            
        Returns:
            Combined loss tensor, or tuple of (combined loss, component losses)
        """
        self.reset()
        weights = self._get_weights()
        
        # Compute individual loss components
        ce_loss = self._compute_ce_loss(logits, labels)
        focal_loss = self._compute_focal_loss(logits, labels)
        soft_f1_loss = self._compute_soft_f1_loss(logits, labels)
        consistency_loss = self._compute_consistency_loss(logits)
        distillation_loss = self._compute_self_distillation(logits, self.temperature)
        
        # Track loss components
        self.losses['ce_loss'] = ce_loss.mean().item()
        self.losses['focal_loss'] = focal_loss.mean().item()
        self.losses['soft_f1_loss'] = soft_f1_loss.mean().item()
        self.losses['consistency_loss'] = consistency_loss.mean().item()
        self.losses['distillation_loss'] = distillation_loss.mean().item()
        
        # Compute metrics
        self.metrics = self.compute_metrics(logits, labels)
        
        # Create component loss dictionary
        component_losses = {
            'ce_loss': ce_loss.mean() * weights['ce_loss'],
            'focal_loss': focal_loss.mean() * weights['focal_loss'],
            'soft_f1_loss': soft_f1_loss.mean() * weights['soft_f1_loss'],
            'consistency_loss': consistency_loss.mean() * weights['consistency_loss'] * self.consistency_weight,
            'distillation_loss': distillation_loss.mean() * weights['distillation_loss'] * self.distillation_weight
        }
        
        # Combine losses with learned weights
        total_loss = sum(component_losses.values())
        
        if return_components:
            return total_loss, component_losses
        return total_loss
    
    def get_loss_info(self) -> Dict[str, Dict[str, float]]:
        """Get tracked losses and metrics."""
        weights = {k: v.item() if hasattr(v, 'item') else v for k, v in self._get_weights().items()}
        return {
            'losses': self.losses,
            'metrics': self.metrics,
            'weights': weights
        }
