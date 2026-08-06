import torch
import torch.nn as nn 
import torch.nn.functional as F

from config import config

use_layernorm = config['use_layernorm']
use_swiglu = config['use_swiglu']

class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048):
        super().__init__()
        pos = torch.arange(max_seq_len, dtype=torch.float)
        theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(pos, theta)
        embedding = torch.cat((angles, angles), dim=-1)
        self.register_buffer('cos', embedding.cos()[None, None, :, :])
        self.register_buffer('sin', embedding.sin()[None, None, :, :])

    def forward(self, x, start_pos=0):
        # x shape: (Batch, n_head, SeqLen, HeadDim)
        # start_pos is the absolute index of x[:, :, 0]: during decoding x holds
        # a single token whose real position is start_pos, not 0.
        seq_len = x.shape[2]
        cos = self.cos[:, :, start_pos:start_pos + seq_len, :]
        sin = self.sin[:, :, start_pos:start_pos + seq_len, :]
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated_half = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (x_rotated_half * sin)

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(x) * gate

class KVCache:
    def __init__(self, n_layer, batch_size, n_kv_head, head_dim, max_len, device, dtype=torch.float32):
        shape = (batch_size, n_kv_head, max_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.max_len = max_len
        self.length = 0 

    def update(self, layer_idx, k, v, start_pos):
        end = start_pos + k.shape[2]
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: {end} > max_len={self.max_len}")
        self.k[layer_idx][:, :, start_pos:end] = k
        self.v[layer_idx][:, :, start_pos:end] = v
        return self.k[layer_idx][:, :, :end], self.v[layer_idx][:, :, :end]

    def reset(self):
        self.length = 0

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, max_len, n_kv_head=None, use_te=False, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.use_te = use_te

        self.n_rep = self.n_head // self.n_kv_head
        
        self.q_size = self.n_head * self.head_dim
        self.kv_size = self.n_kv_head * self.head_dim
        total_qkv_dim = self.q_size + 2 * self.kv_size

        self.rope = RoPE(self.head_dim, max_len)

        ffn_hidden = int(d_model * 8/3) if use_swiglu else int(d_model * 4)

        if use_te:
            import transformer_engine.pytorch as te
            self.ln_attn = te.LayerNormLinear(
                d_model, 
                total_qkv_dim, 
                bias=False,
                normalization="LayerNorm" if use_layernorm else "RMSNorm",
            ) 
            self.c_proj = te.Linear(d_model, d_model, bias=False)
            
            self.ln_mlp = te.LayerNormMLP(
                hidden_size=d_model, 
                ffn_hidden_size=ffn_hidden, 
                bias=False, 
                normalization="LayerNorm" if use_layernorm else "RMSNorm",
                activation='swiglu' if use_swiglu else 'gelu'
            )
        else:
            self.ln1 = nn.LayerNorm(d_model) if use_layernorm else nn.RMSNorm(d_model)
            self.qkv_proj = nn.Linear(d_model, total_qkv_dim, bias=False)
            self.c_proj = nn.Linear(d_model, d_model, bias=False)
            
            self.ln2 = nn.LayerNorm(d_model) if use_layernorm else nn.RMSNorm(d_model)

            if use_swiglu:
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, 2 * ffn_hidden, bias=False),
                    SwiGLU(), 
                    nn.Linear(ffn_hidden, d_model, bias=False)
                )
            else:
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, ffn_hidden, bias=False),
                    nn.GELU(),
                    nn.Linear(ffn_hidden, d_model, bias=False)
                )
                
    def forward(self, x, start_pos=0, kv_cache=None):
        residual = x

        if self.use_te:
            qkv = self.ln_attn(x)
        else:
            x_norm = self.ln1(x)
            qkv = self.qkv_proj(x_norm)

        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=2)
        B, T, _ = q.size()

        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.rope(q, start_pos)
        k = self.rope(k, start_pos)

        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v, start_pos)

        k_len = k.shape[2]
        if T == k_len:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        elif T == 1:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
        else:
            mask = torch.ones(T, k_len, dtype=torch.bool, device=q.device).tril(diagonal=k_len - T)
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        x = residual + self.c_proj(attn_out)

        if self.use_te:
            x = x.contiguous()
            x = x + self.ln_mlp(x)
        else:
            x = x + self.mlp(self.ln2(x))

        return x

class HilbertLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head, max_len, use_te=False, n_kv_head=None):
        super().__init__()
        self.max_len = max_len
        self.use_te = use_te

        self.n_layer = n_layer
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.head_dim = d_model // n_head

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, max_len, use_te=use_te, n_kv_head=n_kv_head, layer_idx=i)
            for i in range(n_layer)
        ])
        
        if use_te:
            import transformer_engine.pytorch as te
            self.final_norm = te.LayerNorm(d_model) if use_layernorm else te.RMSNorm(d_model)
            self.lm_head = te.Linear(d_model, vocab_size, bias=False)
            
        else:
            self.final_norm = nn.LayerNorm(d_model) if use_layernorm else nn.RMSNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
            self.lm_head.weight = self.token_embedding.weight

            self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def new_kv_cache(self, batch_size=1, max_len=None):
        ref = next(self.parameters())
        return KVCache(
            n_layer=self.n_layer,
            batch_size=batch_size,
            n_kv_head=self.n_kv_head,
            head_dim=self.head_dim,
            max_len=max_len if max_len is not None else self.max_len,
            device=ref.device,
            dtype=ref.dtype,
        )

    def forward(self, x, targets=None, kv_cache=None, start_pos=None):
        if start_pos is None:
            start_pos = kv_cache.length if kv_cache is not None else 0

        x = self.token_embedding(x)

        for layer in self.layers:
            x = layer(x, start_pos=start_pos, kv_cache=kv_cache)

        if kv_cache is not None:
            kv_cache.length = start_pos + x.shape[1]
            if targets is None:
                x = x[:, -1:, :]

        x = self.final_norm(x)

        if self.use_te:
            import transformer_engine.pytorch as te
            with te.fp8_autocast(enabled=False):
                logits = self.lm_head(x)
        else:
            logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            return logits, loss

        return logits
    

