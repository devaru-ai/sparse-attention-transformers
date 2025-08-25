import torch
import torch.nn as nn
import math

class LSHAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, bucket_size=64, n_hashes=8, causal=False, max_bucket_size=256):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self.causal = causal
        self.max_bucket_size = max_bucket_size

        self.qk_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def hash_vectors(self, qk):
        batch, heads, seqlen, head_dim = qk.shape
        n_buckets = max(2, 2 * (seqlen // (2 * self.bucket_size)))
        # always at least 2 buckets to avoid empty dimension
        rot_size = max(1, n_buckets // 2)
        rotations_shape = (self.n_hashes, self.num_heads, head_dim, rot_size)
        random_rotations = torch.randn(rotations_shape, device=qk.device)
        rotated = torch.einsum('bhnd,rhdc->bhnrc', qk, random_rotations)
        rotated = torch.cat([rotated, -rotated], dim=-1)
        buckets = torch.argmax(rotated, dim=-1)
        return buckets, n_buckets


    def forward(self, x):
        batch, seqlen, d_model = x.size()
        num_heads, head_dim = self.num_heads, self.head_dim
        qk = self.qk_proj(x).view(batch, seqlen, num_heads, head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(batch, seqlen, num_heads, head_dim).permute(0, 2, 1, 3)

        buckets, n_buckets = self.hash_vectors(qk)
        out = torch.zeros_like(qk)

        for h in range(self.n_hashes):
            bucket = buckets[..., h]
            # Vectorized grouping by bucket
            for b_ix in range(n_buckets):
                mask = (bucket == b_ix)
                if not mask.any():
                    continue
                indices = mask.nonzero(as_tuple=False)
                b_batch, b_head, b_pos = indices[:,0], indices[:,1], indices[:,2]
                qk_sel = qk[b_batch, b_head, b_pos]
                v_sel  = v[b_batch, b_head, b_pos]
                # Cap bucket size for efficiency
                if qk_sel.size(0) > self.max_bucket_size:
                    print(f"WARNING: Bucket {b_ix} too large ({qk_sel.size(0)} items), capping to {self.max_bucket_size}")
                    idx = torch.randperm(qk_sel.size(0))[:self.max_bucket_size]
                    qk_sel = qk_sel[idx]
                    v_sel = v_sel[idx]
                    b_batch = b_batch[idx]
                    b_head = b_head[idx]
                    b_pos = b_pos[idx]
                # Weighted attention chunk
                dots = torch.matmul(qk_sel, qk_sel.transpose(0,1)) / math.sqrt(self.head_dim)
                if self.causal:
                    relpos = b_pos.unsqueeze(-1) - b_pos.unsqueeze(0)
                    dots[relpos < 0] = -float('inf')
                weights = torch.softmax(dots, dim=-1)
                out_sel = torch.matmul(weights, v_sel)
                # Scatter result back
                out[b_batch, b_head, b_pos] += out_sel

        out = out / self.n_hashes
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, seqlen, d_model)
        return self.out_proj(out)
