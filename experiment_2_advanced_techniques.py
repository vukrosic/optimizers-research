#!/usr/bin/env python3
"""
🔬 EXPERIMENT 2: ADVANCED TRAINING DYNAMICS & REGULARIZATION
===============================================================================

Building on our successful Muon optimizer research, this experiment explores:

1. ADVANCED REGULARIZATION TECHNIQUES:
   - Stochastic Depth (Layer Dropout)
   - DropPath (Structured Dropout)
   - Gradient Noise Injection
   - Weight Noise (Gaussian)
   - Spectral Normalization

2. ADAPTIVE TRAINING DYNAMICS:
   - Learning Rate Range Test
   - Cyclic Learning Rates
   - Gradient Accumulation Scheduling
   - Batch Size Adaptation
   - Early Stopping with Patience

3. ARCHITECTURE ENHANCEMENTS:
   - Attention Improvements (Multi-Query, Grouped-Query)
   - Better Weight Initialization (Xavier, He, Orthogonal)
   - Layer Scale
   - Pre-Norm vs Post-Norm comparison

4. ADVANCED LOSS TECHNIQUES:
   - Contrastive Loss Components
   - Knowledge Distillation Loss
   - Auxiliary Task Losses
   - Sharpness-Aware Minimization (SAM)

Goal: Push beyond 4.68 validation loss achieved by Muon baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
from tqdm import tqdm
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import our previous successful components
from llm import (
    set_seed, Muon, zeropower_via_newtonschulz5, Lion, FocalLoss, 
    LabelSmoothingCrossEntropy, MoEModelConfig, load_and_cache_data,
    TextTokenDataset, evaluate_model
)

@dataclass
class AdvancedExperimentConfig(MoEModelConfig):
    """Extended configuration for advanced techniques"""
    
    # Experiment identification
    experiment_name: str = "Advanced_Techniques"
    
    # Advanced regularization
    stochastic_depth_rate: float = 0.1  # Probability of dropping layers
    droppath_rate: float = 0.1  # Structured dropout rate
    gradient_noise_std: float = 0.01  # Gradient noise injection
    weight_noise_std: float = 0.001  # Weight noise during training
    use_spectral_norm: bool = False  # Spectral normalization
    
    # Training dynamics
    use_cyclic_lr: bool = False  # Cyclic learning rates
    lr_cycle_length: int = 100  # Steps per cycle
    use_lr_range_test: bool = False  # Learning rate range test
    adaptive_grad_accum: bool = False  # Adaptive gradient accumulation
    early_stopping_patience: int = 50  # Early stopping patience
    
    # Architecture enhancements
    attention_type: str = "standard"  # "standard", "multi_query", "grouped_query"
    init_method: str = "xavier"  # "xavier", "he", "orthogonal"
    use_layer_scale: bool = False  # Layer scale technique
    norm_position: str = "pre"  # "pre" or "post" normalization
    
    # Advanced loss techniques
    use_sam: bool = False  # Sharpness-Aware Minimization
    sam_rho: float = 0.05  # SAM perturbation radius
    use_knowledge_distillation: bool = False  # Knowledge distillation
    kd_temperature: float = 4.0  # Distillation temperature
    kd_alpha: float = 0.7  # Distillation loss weight


class StochasticDepth(nn.Module):
    """Stochastic Depth - randomly drop entire transformer blocks"""
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x, residual):
        if not self.training:
            return x + residual
        
        # Randomly drop the block output
        if random.random() < self.drop_prob:
            return residual  # Skip the block entirely
        else:
            return x + residual


class DropPath(nn.Module):
    """DropPath - structured dropout that drops entire samples in batch"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Work with diff dim tensors
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output


class LayerScale(nn.Module):
    """Layer Scale - learnable scaling for residual connections"""
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return self.gamma * x


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention - shared key/value across heads for efficiency"""
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Multi-query: one key/value, multiple queries
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, self.d_k, bias=False)  # Single key
        self.w_v = nn.Linear(d_model, self.d_k, bias=False)  # Single value
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        
        # Multi-query attention
        Q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, D)
        K = self.w_k(x).view(B, T, 1, self.d_k).expand(-1, -1, self.n_heads, -1).transpose(1, 2)  # (B, H, T, D)
        V = self.w_v(x).view(B, T, 1, self.d_k).expand(-1, -1, self.n_heads, -1).transpose(1, 2)  # (B, H, T, D)

        # Scaled dot-product attention with causal mask
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.w_o(attn_output)


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization - finds flatter minima for better generalization"""
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # Climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # Get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # Do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # The closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # Put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(dtype=torch.float32)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    dtype=torch.float32
               )
        return norm.to(shared_device)


class CyclicLR(torch.optim.lr_scheduler._LRScheduler):
    """Cyclic Learning Rate scheduler"""
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000, mode='triangular', gamma=1.0):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_up
        self.mode = mode
        self.gamma = gamma
        
        super().__init__(optimizer)

    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / (self.step_size_up + self.step_size_down))
        x = abs(self.last_epoch / self.step_size_up - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        elif self.mode == 'triangular2':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) / float(2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * (self.gamma ** self.last_epoch)
        
        return [lr for _ in self.optimizer.param_groups]


def inject_gradient_noise(model, noise_std: float):
    """Inject Gaussian noise into gradients for regularization"""
    if noise_std <= 0:
        return
    
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_std
            param.grad.add_(noise)


def inject_weight_noise(model, noise_std: float):
    """Inject noise into weights during training"""
    if noise_std <= 0:
        return
    
    for param in model.parameters():
        if param.requires_grad:
            noise = torch.randn_like(param) * noise_std
            param.data.add_(noise)


def apply_spectral_normalization(model):
    """Apply spectral normalization to linear layers"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module = nn.utils.spectral_norm(module)


def initialize_weights(model, method: str = "xavier"):
    """Advanced weight initialization"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if method == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif method == "he":
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif method == "orthogonal":
                nn.init.orthogonal_(module.weight)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)


class AdvancedTransformerBlock(nn.Module):
    """Enhanced transformer block with advanced techniques"""
    def __init__(self, config: AdvancedExperimentConfig):
        super().__init__()
        self.config = config
        
        # Choose attention mechanism
        if config.attention_type == "multi_query":
            self.attention = MultiQueryAttention(
                config.d_model, config.n_heads, config.max_seq_len, config.dropout
            )
        else:
            # Standard multi-head attention
            self.attention = nn.MultiheadAttention(
                config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
            )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Normalization layers
        self.norm1 = nn.RMSNorm(config.d_model)
        self.norm2 = nn.RMSNorm(config.d_model)
        
        # Advanced techniques
        self.stochastic_depth = StochasticDepth(config.stochastic_depth_rate)
        self.droppath = DropPath(config.droppath_rate)
        
        if config.use_layer_scale:
            self.layer_scale1 = LayerScale(config.d_model)
            self.layer_scale2 = LayerScale(config.d_model)

    def forward(self, x):
        # Pre-norm or post-norm
        if self.config.norm_position == "pre":
            # Pre-norm (standard for modern transformers)
            residual = x
            x_norm = self.norm1(x)
            
            if hasattr(self.attention, 'forward'):
                attn_out = self.attention(x_norm)
            else:
                attn_mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
                attn_out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=attn_mask)
            
            if self.config.use_layer_scale:
                attn_out = self.layer_scale1(attn_out)
            
            attn_out = self.droppath(attn_out)
            x = self.stochastic_depth(attn_out, residual)
            
            # Feed-forward
            residual = x
            x_norm = self.norm2(x)
            ff_out = self.feed_forward(x_norm)
            
            if self.config.use_layer_scale:
                ff_out = self.layer_scale2(ff_out)
            
            ff_out = self.droppath(ff_out)
            x = self.stochastic_depth(ff_out, residual)
            
        else:
            # Post-norm (original transformer)
            residual = x
            if hasattr(self.attention, 'forward'):
                attn_out = self.attention(x)
            else:
                attn_mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
                attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
            
            x = self.norm1(self.stochastic_depth(attn_out, residual))
            
            residual = x
            ff_out = self.feed_forward(x)
            x = self.norm2(self.stochastic_depth(ff_out, residual))
        
        return x


class AdvancedLLM(nn.Module):
    """Advanced LLM with experimental techniques"""
    def __init__(self, config: AdvancedExperimentConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            AdvancedTransformerBlock(config)
            for _ in range(config.n_layers)
        ])
        
        # Output layers
        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        # Apply advanced initialization
        initialize_weights(self, config.init_method)
        
        # Apply spectral normalization if requested
        if config.use_spectral_norm:
            apply_spectral_normalization(self)

    def forward(self, x):
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Output
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        
        return logits


def setup_advanced_optimizer(model, config: AdvancedExperimentConfig):
    """Setup optimizer with SAM if requested"""
    # Separate parameters
    matrix_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            matrix_params.append(param)
        else:
            other_params.append(param)
    
    print(f"  Matrix parameters: {sum(p.numel() for p in matrix_params):,}")
    print(f"  Other parameters: {sum(p.numel() for p in other_params):,}")
    
    # Create base optimizers
    if matrix_params:
        matrix_opt = Muon(matrix_params, lr=config.muon_lr, momentum=0.95)
    if other_params:
        other_opt = torch.optim.AdamW(other_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)
    
    base_optimizers = []
    if matrix_params:
        base_optimizers.append(matrix_opt)
    if other_params:
        base_optimizers.append(other_opt)
    
    # Wrap with SAM if requested
    if config.use_sam:
        sam_optimizers = []
        for opt in base_optimizers:
            sam_opt = SAM(opt.param_groups, lambda params: type(opt)(params), rho=config.sam_rho)
            sam_optimizers.append(sam_opt)
        return sam_optimizers
    
    return base_optimizers


def setup_advanced_scheduler(optimizers, config: AdvancedExperimentConfig):
    """Setup advanced learning rate schedulers"""
    schedulers = []
    
    for optimizer in optimizers:
        if config.use_cyclic_lr:
            # Cyclic learning rate
            scheduler = CyclicLR(
                optimizer, 
                base_lr=config.muon_lr * 0.1,
                max_lr=config.muon_lr,
                step_size_up=config.lr_cycle_length
            )
        else:
            # Standard cosine with warmup
            warmup_steps = config.max_steps // 20
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        schedulers.append(scheduler)
    
    return schedulers


class ExperimentTracker:
    """Enhanced experiment tracking"""
    def __init__(self):
        self.experiments = []
        self.current_experiment = None

    def start_experiment(self, name, config):
        self.current_experiment = {
            'name': name,
            'config': config,
            'metrics': [],
            'start_time': time.time(),
            'best_loss': float('inf'),
            'early_stopped': False,
            'convergence_step': None
        }

    def log_metrics(self, step, loss, accuracy, lr=None):
        if self.current_experiment:
            self.current_experiment['metrics'].append({
                'step': step,
                'loss': loss,
                'accuracy': accuracy,
                'lr': lr,
                'timestamp': time.time()
            })
            
            # Track best loss and convergence
            if loss < self.current_experiment['best_loss']:
                self.current_experiment['best_loss'] = loss
                if self.current_experiment['convergence_step'] is None and loss < 5.0:
                    self.current_experiment['convergence_step'] = step

    def end_experiment(self, final_metrics, early_stopped=False):
        if self.current_experiment:
            self.current_experiment['end_time'] = time.time()
            self.current_experiment['duration'] = (
                self.current_experiment['end_time'] - self.current_experiment['start_time']
            )
            self.current_experiment['final_metrics'] = final_metrics
            self.current_experiment['early_stopped'] = early_stopped
            self.experiments.append(self.current_experiment)
            self.current_experiment = None

    def compare_experiments(self):
        """Enhanced experiment comparison"""
        if not self.experiments:
            return None
        
        best_exp = min(self.experiments, key=lambda x: x['final_metrics']['val_loss'])
        
        print(f"\n🏆 ADVANCED EXPERIMENT COMPARISON:")
        print(f"{'='*80}")
        
        # Sort by performance
        sorted_experiments = sorted(self.experiments, key=lambda x: x['final_metrics']['val_loss'])
        
        for i, exp in enumerate(sorted_experiments):
            status = "🥇 BEST" if exp == best_exp else f"#{i+1}"
            
            print(f"{status} {exp['name']}")
            print(f"   Final Loss: {exp['final_metrics']['val_loss']:.4f}")
            print(f"   Best Loss: {exp['best_loss']:.4f}")
            print(f"   Final Accuracy: {exp['final_metrics']['val_accuracy']:.4f}")
            print(f"   Training Time: {exp['duration']/60:.1f} min")
            print(f"   Convergence Step: {exp['convergence_step'] or 'N/A'}")
            print(f"   Early Stopped: {'Yes' if exp['early_stopped'] else 'No'}")
            
            # Highlight key techniques
            config = exp['config']
            techniques = []
            if config.stochastic_depth_rate > 0:
                techniques.append(f"StochDepth({config.stochastic_depth_rate})")
            if config.use_sam:
                techniques.append(f"SAM({config.sam_rho})")
            if config.use_cyclic_lr:
                techniques.append("CyclicLR")
            if config.attention_type != "standard":
                techniques.append(f"{config.attention_type.upper()}")
            if config.use_layer_scale:
                techniques.append("LayerScale")
            
            if techniques:
                print(f"   Techniques: {', '.join(techniques)}")
            print()
        
        return best_exp


def train_advanced_model(config: AdvancedExperimentConfig, train_loader, val_loader, tracker):
    """Train model with advanced techniques"""
    print(f"\n🚀 Training Advanced Model: {config.experiment_name}")
    print(f"   Techniques: ", end="")
    
    techniques = []
    if config.stochastic_depth_rate > 0:
        techniques.append(f"StochDepth({config.stochastic_depth_rate})")
    if config.use_sam:
        techniques.append(f"SAM({config.sam_rho})")
    if config.use_cyclic_lr:
        techniques.append("CyclicLR")
    if config.attention_type != "standard":
        techniques.append(config.attention_type.upper())
    
    print(", ".join(techniques) if techniques else "Baseline+")
    
    tracker.start_experiment(config.experiment_name, config)
    
    # Initialize model
    set_seed(42)
    model = AdvancedLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")
    
    # Setup optimizers and schedulers
    optimizers = setup_advanced_optimizer(model, config)
    schedulers = setup_advanced_scheduler(optimizers, config)
    
    scaler = GradScaler() if config.use_amp else None
    
    # Training loop with advanced techniques
    model.train()
    step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    pbar = tqdm(total=config.max_steps, desc=f"Training {config.experiment_name}")
    
    def closure():
        """Closure for SAM optimizer"""
        optimizers[0].zero_grad()
        # This would be called by SAM for the second forward pass
        return None
    
    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break
            
            x, y = x.to(device), y.to(device)
            
            # Inject weight noise if specified
            if config.weight_noise_std > 0:
                inject_weight_noise(model, config.weight_noise_std)
            
            # Forward pass
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
            
            # Inject gradient noise
            if config.gradient_noise_std > 0:
                inject_gradient_noise(model, config.gradient_noise_std)
            
            # Optimizer step
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_sam:
                    # SAM requires special handling
                    if config.use_amp:
                        for optimizer in optimizers:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        
                        for optimizer in optimizers:
                            scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        for optimizer in optimizers:
                            optimizer.step(closure)
                else:
                    # Standard optimization
                    if config.use_amp:
                        for optimizer in optimizers:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        
                        for optimizer in optimizers:
                            scaler.step(optimizer)
                            optimizer.zero_grad()
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        for optimizer in optimizers:
                            optimizer.step()
                            optimizer.zero_grad()
                
                # Update schedulers
                for scheduler in schedulers:
                    scheduler.step()
            
            # Logging
            if step % 50 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = loss.item() * config.gradient_accumulation_steps
                    current_lr = optimizers[0].param_groups[0]['lr']
                    
                    tracker.log_metrics(step, current_loss, accuracy, current_lr)
                
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'lr': f'{current_lr:.2e}',
                    'best': f'{best_val_loss:.4f}'
                })
            
            # Evaluation and early stopping
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                current_val_loss = eval_metrics['val_loss']
                
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    print(f"\n🎯 Step {step}: New best validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= config.early_stopping_patience:
                    print(f"\n⏹️ Early stopping at step {step} (patience: {patience_counter})")
                    break
            
            step += 1
            if step % 50 == 0:
                pbar.update(50)
    
    pbar.close()
    
    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    early_stopped = patience_counter >= config.early_stopping_patience
    tracker.end_experiment(final_eval, early_stopped)
    
    return model, final_eval


def run_advanced_experiment():
    """Run comprehensive advanced techniques experiment"""
    print(f"🔬 EXPERIMENT 2: ADVANCED TRAINING TECHNIQUES")
    print(f"{'='*80}")
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Load data (reuse from previous experiment)
    temp_config = MoEModelConfig()
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    vocab_size = temp_config.vocab_size
    
    dataset = TextTokenDataset(tokens, temp_config.max_seq_len)
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=2)
    
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    
    # Define advanced experiments
    experiments = [
        # Baseline (reproduce our best result)
        AdvancedExperimentConfig(
            experiment_name="Baseline_Muon",
            d_model=384, n_heads=8, n_layers=6, d_ff=1536,
            vocab_size=vocab_size, max_steps=500,
            stochastic_depth_rate=0.0, use_sam=False
        ),
        
        # Stochastic Depth
        AdvancedExperimentConfig(
            experiment_name="StochasticDepth",
            d_model=384, n_heads=8, n_layers=6, d_ff=1536,
            vocab_size=vocab_size, max_steps=500,
            stochastic_depth_rate=0.15, droppath_rate=0.1
        ),
        
        # SAM (Sharpness-Aware Minimization)
        AdvancedExperimentConfig(
            experiment_name="SAM_Optimizer",
            d_model=384, n_heads=8, n_layers=6, d_ff=1536,
            vocab_size=vocab_size, max_steps=500,
            use_sam=True, sam_rho=0.05
        ),
        
        # Multi-Query Attention
        AdvancedExperimentConfig(
            experiment_name="MultiQuery_Attention",
            d_model=384, n_heads=8, n_layers=6, d_ff=1536,
            vocab_size=vocab_size, max_steps=500,
            attention_type="multi_query", use_layer_scale=True
        ),
        
        # Cyclic Learning Rate
        AdvancedExperimentConfig(
            experiment_name="CyclicLR",
            d_model=384, n_heads=8, n_layers=6, d_ff=1536,
            vocab_size=vocab_size, max_steps=500,
            use_cyclic_lr=True, lr_cycle_length=100
        ),
        
        # Combined Advanced Techniques
        AdvancedExperimentConfig(
            experiment_name="Combined_Advanced",
            d_model=384, n_heads=8, n_layers=6, d_ff=1536,
            vocab_size=vocab_size, max_steps=500,
            stochastic_depth_rate=0.1, droppath_rate=0.05,
            use_layer_scale=True, attention_type="multi_query",
            gradient_noise_std=0.005, use_cyclic_lr=True
        )
    ]
    
    # Run experiments
    for i, config in enumerate(experiments):
        print(f"\n🧪 ADVANCED EXPERIMENT {i+1}/{len(experiments)}: {config.experiment_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        model, final_metrics = train_advanced_model(config, train_loader, val_loader, tracker)
        total_time = time.time() - start_time
        
        print(f"\n📊 Results:")
        print(f"   Training time: {total_time/60:.1f} minutes")
        print(f"   Final Loss: {final_metrics['val_loss']:.4f}")
        print(f"   Final Accuracy: {final_metrics['val_accuracy']:.4f}")
    
    # Compare all experiments
    best_experiment = tracker.compare_experiments()
    
    print(f"\n🎯 EXPERIMENT 2 CONCLUSION:")
    print(f"{'='*80}")
    if best_experiment:
        print(f"✨ Best approach: {best_experiment['name']}")
        print(f"📈 Achieved validation loss: {best_experiment['final_metrics']['val_loss']:.4f}")
        print(f"⚡ Training time: {best_experiment['duration']/60:.1f} minutes")
        
        # Compare with our original best (4.6801)
        original_best = 4.6801
        improvement = (original_best - best_experiment['final_metrics']['val_loss']) / original_best * 100
        if improvement > 0:
            print(f"🚀 Improvement over original best: {improvement:.1f}%")
        else:
            print(f"📊 Performance vs original: {-improvement:.1f}% difference")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    run_advanced_experiment()
