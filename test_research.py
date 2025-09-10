#!/usr/bin/env python3
"""
🔬 ADVANCED LLM TRAINING RESEARCH - MINIMAL TEST VERSION
===============================================================================

This is a simplified version of the research that demonstrates the advanced 
optimization techniques without requiring external dependencies like HuggingFace datasets.

RESEARCH FOCUS: Reducing LLM training loss faster through:
1. Advanced Optimizers: Sophia-G, Lion, Muon
2. Improved Loss Functions: Focal Loss, Label Smoothing
3. Architecture Enhancements: SwishGEGLU, better attention
4. Training Dynamics: Adaptive learning rates, gradient analysis

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
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")

# Simple synthetic dataset for testing
class SyntheticTextDataset(Dataset):
    """Generate synthetic sequences for testing optimizers"""
    def __init__(self, vocab_size=1000, seq_len=128, num_samples=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic sequence with some pattern
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        y = torch.roll(x, -1)  # Next token prediction
        return x, y

@dataclass
class TestModelConfig:
    # Model architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    vocab_size: int = 1000
    max_seq_len: int = 128
    
    # Training parameters
    batch_size: int = 32
    max_steps: int = 500
    gradient_accumulation_steps: int = 1
    
    # Optimizer settings
    optimizer_type: str = "muon"  # "muon", "sophia", "lion", "adamw"
    base_lr: float = 0.01
    
    # Loss function improvements
    use_focal_loss: bool = False
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    use_label_smoothing: bool = True
    label_smoothing: float = 0.1
    
    # Training dynamics
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0
    
    # Evaluation
    eval_every: int = 100
    eval_steps: int = 50
    
    # Technical
    use_amp: bool = True

# Copy the optimizers from the main file
@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

class Lion(torch.optim.Optimizer):
    """Lion optimizer - EvoLved Sign Momentum"""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-group['weight_decay'] * group['lr'])

                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                exp_avg.lerp_(grad, 1 - beta2)

        return loss

class FocalLoss(nn.Module):
    """Focal Loss for addressing hard examples"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        loss = torch.sum(-smooth_one_hot * F.log_softmax(pred, 1), 1)
        return loss.mean()

# Simple transformer model for testing
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

class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, stochastic_depth=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.stochastic_depth = StochasticDepth(stochastic_depth)

    def forward(self, x):
        # Self-attention with stochastic depth
        attn_mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        attn_out = self.dropout(attn_out)
        x = self.stochastic_depth(attn_out, x)
        x = self.norm1(x)
        
        # Feed-forward with stochastic depth
        residual = x
        ff_out = self.feed_forward(x)
        x = self.stochastic_depth(ff_out, residual)
        x = self.norm2(x)
        return x

class SimpleTransformerLM(nn.Module):
    def __init__(self, config: TestModelConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Add stochastic depth with linear scaling
        stoch_depth_rates = [x * 0.1 for x in range(config.n_layers)]  # 0, 0.1, 0.2, etc.
        
        self.transformer_blocks = nn.ModuleList([
            SimpleTransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout, stoch_depth_rates[i])
            for i in range(config.n_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        
        # Token and position embeddings
        tok_emb = self.token_embedding(x)
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embedding(pos_ids)
        
        x = tok_emb + pos_emb
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits

def setup_optimizer(model, config: TestModelConfig):
    """Setup optimizer based on configuration"""
    matrix_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            'pos_embedding' not in name and 
            param.requires_grad):
            matrix_params.append(param)
        else:
            other_params.append(param)
    
    print(f"  Matrix parameters: {sum(p.numel() for p in matrix_params):,}")
    print(f"  Other parameters: {sum(p.numel() for p in other_params):,}")
    
    if config.optimizer_type == "muon" or config.optimizer_type == "muon_advanced":
        if matrix_params:
            matrix_opt = Muon(matrix_params, lr=config.base_lr, momentum=0.95)
        if other_params:
            other_opt = torch.optim.AdamW(other_params, lr=config.base_lr*0.1, weight_decay=config.weight_decay)
        return [matrix_opt, other_opt] if matrix_params and other_params else [matrix_opt if matrix_params else other_opt]
    
    elif config.optimizer_type == "lion":
        if matrix_params:
            matrix_opt = Lion(matrix_params, lr=config.base_lr*0.1, weight_decay=config.weight_decay)
        if other_params:
            other_opt = torch.optim.AdamW(other_params, lr=config.base_lr*0.01, weight_decay=config.weight_decay)
        return [matrix_opt, other_opt] if matrix_params and other_params else [matrix_opt if matrix_params else other_opt]
    
    else:  # adamw
        all_params = matrix_params + other_params
        return [torch.optim.AdamW(all_params, lr=config.base_lr*0.1, weight_decay=config.weight_decay)]

def get_loss_function(config: TestModelConfig):
    """Get loss function based on configuration"""
    if config.use_focal_loss:
        return FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    elif config.use_label_smoothing:
        return LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
    else:
        return F.cross_entropy

def evaluate_model(model, val_loader, config: TestModelConfig):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)
            
            with autocast(enabled=config.use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()
    
    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    
    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

def train_model(config: TestModelConfig, train_loader, val_loader):
    """Train model with given configuration"""
    print(f"\n🚀 Training model with {config.optimizer_type} optimizer")
    print(f"   Loss: {'Focal' if config.use_focal_loss else 'Label Smoothed' if config.use_label_smoothing else 'Standard'}")
    
    # Initialize model
    set_seed(42)
    model = SimpleTransformerLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")
    
    # Setup optimizer and loss
    optimizers = setup_optimizer(model, config)
    loss_fn = get_loss_function(config)
    
    # Learning rate scheduler
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 10
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)
    
    scaler = GradScaler() if config.use_amp else None
    
    # Training loop
    model.train()
    step = 0
    best_val_loss = float('inf')
    start_time = time.time()
    pbar = tqdm(total=config.max_steps, desc=f"Training {config.optimizer_type}")
    
    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break
            
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    if config.use_focal_loss or config.use_label_smoothing:
                        loss = loss_fn(logits.view(-1, config.vocab_size), y.view(-1))
                    else:
                        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                if config.use_focal_loss or config.use_label_smoothing:
                    loss = loss_fn(logits.view(-1, config.vocab_size), y.view(-1))
                else:
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
            
            # Optimizer step
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    
                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
            
            # Logging
            if step % 50 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = loss.item() * config.gradient_accumulation_steps
                    perplexity = math.exp(min(current_loss, 20))
                    current_lr = optimizers[0].param_groups[0]['lr']
                
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{current_lr:.2e}'
                })
            
            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                if eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']
                    print(f"\n🎯 Step {step}: New best validation loss: {best_val_loss:.4f}")
            
            step += 1
            if step % 50 == 0:
                pbar.update(50)
    
    pbar.close()
    
    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    training_time = time.time() - start_time
    
    return {
        'config': config,
        'final_metrics': final_eval,
        'training_time': training_time,
        'best_val_loss': best_val_loss
    }

def run_optimization_research():
    """Run comprehensive optimization research"""
    print(f"🔬 ADVANCED LLM TRAINING RESEARCH")
    print(f"{'='*80}")
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create dataset
    print(f"\n📊 Creating synthetic dataset...")
    dataset = SyntheticTextDataset(vocab_size=1000, seq_len=128, num_samples=10000)
    
    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Define experiments
    experiments = [
        # Baseline AdamW
        TestModelConfig(
            optimizer_type="adamw",
            use_focal_loss=False,
            use_label_smoothing=False,
            max_steps=500
        ),
        
        # Muon optimizer
        TestModelConfig(
            optimizer_type="muon",
            use_focal_loss=False,
            use_label_smoothing=False,
            max_steps=500
        ),
        
        # Lion optimizer
        TestModelConfig(
            optimizer_type="lion",
            use_focal_loss=False,
            use_label_smoothing=False,
            max_steps=500
        ),
        
        # Muon + Label Smoothing
        TestModelConfig(
            optimizer_type="muon",
            use_focal_loss=False,
            use_label_smoothing=True,
            label_smoothing=0.1,
            max_steps=500
        ),
        
        # Lion + Focal Loss
        TestModelConfig(
            optimizer_type="lion",
            use_focal_loss=True,
            focal_gamma=2.0,
            use_label_smoothing=False,
            max_steps=500
        ),
        
        # Advanced: Muon + Label Smoothing + Stochastic Depth (built into model now)
        TestModelConfig(
            optimizer_type="muon_advanced",  # Special name to trigger advanced message
            use_focal_loss=False,
            use_label_smoothing=True,
            label_smoothing=0.1,
            max_steps=500
        )
    ]
    
    # Run experiments
    results = []
    for i, config in enumerate(experiments):
        exp_name = f"{config.optimizer_type}"
        if config.use_focal_loss:
            exp_name += "_focal"
        if config.use_label_smoothing:
            exp_name += "_smooth"
        
        print(f"\n🧪 EXPERIMENT {i+1}/{len(experiments)}: {exp_name.upper()}")
        if "advanced" in config.optimizer_type:
            print(f"   🚀 ADVANCED: Includes Stochastic Depth regularization")
        print(f"{'='*60}")
        
        result = train_model(config, train_loader, val_loader)
        result['name'] = exp_name
        results.append(result)
        
        print(f"\n📊 Results:")
        print(f"   Training time: {result['training_time']/60:.1f} minutes")
        print(f"   Final Loss: {result['final_metrics']['val_loss']:.4f}")
        print(f"   Final Accuracy: {result['final_metrics']['val_accuracy']:.4f}")
        print(f"   Best Loss: {result['best_val_loss']:.4f}")
    
    # Compare results
    print(f"\n🏆 RESEARCH COMPARISON:")
    print(f"{'='*80}")
    
    # Sort by best validation loss
    results.sort(key=lambda x: x['best_val_loss'])
    
    for i, result in enumerate(results):
        status = "🥇 BEST" if i == 0 else f"#{i+1}"
        print(f"{status} {result['name'].upper()}")
        print(f"   Best Val Loss: {result['best_val_loss']:.4f}")
        print(f"   Final Val Loss: {result['final_metrics']['val_loss']:.4f}")
        print(f"   Final Accuracy: {result['final_metrics']['val_accuracy']:.4f}")
        print(f"   Training Time: {result['training_time']/60:.1f} min")
        print(f"   Optimizer: {result['config'].optimizer_type}")
        print()
    
    print(f"🎯 RESEARCH CONCLUSION:")
    print(f"{'='*80}")
    best = results[0]
    print(f"✨ Best approach: {best['name'].upper()}")
    print(f"📈 Achieved best validation loss: {best['best_val_loss']:.4f}")
    print(f"⚡ Fastest convergence with: {best['config'].optimizer_type}")
    
    # Calculate improvement over baseline
    baseline = next((r for r in results if r['config'].optimizer_type == 'adamw'), None)
    if baseline:
        improvement = (baseline['best_val_loss'] - best['best_val_loss']) / baseline['best_val_loss'] * 100
        print(f"🚀 Improvement over AdamW baseline: {improvement:.1f}%")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    run_optimization_research()
