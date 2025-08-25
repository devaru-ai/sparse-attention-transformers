import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableLSHAttention(nn.Module):
    """
    LSH Attention with a trainable hash projection matrix and a capped bucket size for speed.
    Provides a regularizer loss for bucket coherence.
    """
    def __init__(self, d_model, num_heads=8, bucket_size=64, n_hashes=4, max_bucket_size=128):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self.max_bucket_size = max_bucket_size
        self.hash_proj = nn.Parameter(torch.randn(num_heads, self.head_dim, bucket_size))
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, router=None, regularizer=False):
        print(f"[LearnableLSHAttention] input x shape: {x.shape}")
        batch, seqlen, d_model = x.size()

        q_proj_out = self.q_proj(x)
        k_proj_out = self.k_proj(x)
        v_proj_out = self.v_proj(x)
        print(f"  proj shapes: {q_proj_out.shape}, {k_proj_out.shape}, {v_proj_out.shape}")

        # View: (batch, seqlen, num_heads, head_dim)
        q = q_proj_out.view(batch, seqlen, self.num_heads, self.head_dim).permute(0,2,1,3).contiguous()
        k = k_proj_out.view(batch, seqlen, self.num_heads, self.head_dim).permute(0,2,1,3).contiguous()
        v = v_proj_out.view(batch, seqlen, self.num_heads, self.head_dim).permute(0,2,1,3).contiguous()
        print(f"  q, k, v shapes: {q.shape}, {k.shape}, {v.shape}")

        hash_scores = torch.einsum('bhnd,hdf->bhnf', q, self.hash_proj)
        print(f"  hash_scores shape: {hash_scores.shape}")
        buckets = hash_scores.argmax(dim=-1)
        print(f"  buckets shape: {buckets.shape}")

        reg_loss = None
        if regularizer:
            print("  Computing regularization loss...")
            for h in range(self.num_heads):
                q_states = q[:,h,:,:]
                sim = F.cosine_similarity(q_states.unsqueeze(2), q_states.unsqueeze(1), dim=-1)
                bucket_eq = (buckets[:,h,:].unsqueeze(2) == buckets[:,h,:].unsqueeze(1)).float()
                reg_loss = (sim * bucket_eq - sim * (1-bucket_eq)).mean() + (reg_loss if reg_loss is not None else 0)
            print(f"  reg_loss value: {reg_loss}")

        out = torch.zeros_like(q)
        for h in range(self.num_heads):
            for b_ix in range(self.bucket_size):
                mask = (buckets[:,h,:] == b_ix)
                if not mask.any(): continue
                idx = mask.nonzero(as_tuple=False)
                b_idx, t_idx = idx[:,0], idx[:,1]
                bucket_size_now = len(b_idx)
                if bucket_size_now > self.max_bucket_size:
                    print(f"WARNING: Bucket {b_ix} (head {h}) size {bucket_size_now} > {self.max_bucket_size}. Truncating for speed.")
                    select = torch.randperm(bucket_size_now)[:self.max_bucket_size]
                    b_idx = b_idx[select]
                    t_idx = t_idx[select]
                q_sel = q[b_idx, h, t_idx]
                k_sel = k[b_idx, h, t_idx]
                v_sel = v[b_idx, h, t_idx]
                dots = torch.matmul(q_sel, k_sel.t()) / (self.head_dim ** 0.5)
                attn = torch.softmax(dots, dim=-1)
                out_sel = torch.matmul(attn, v_sel)
                out[b_idx, h, t_idx] = out_sel

        print(f"  out (bucketed attention) shape: {out.shape}")

        if router is not None:
            print("  Applying DynamicLocalGlobalRouter...")
            out = router(q, k, v, out, buckets)
            print(f"  out (after router) shape: {out.shape}")

        out = out.permute(0,2,1,3).contiguous().view(batch, seqlen, -1)
        print(f"  out (final attention) shape: {out.shape}")

        out = self.out_proj(out)
        print(f"  out (projected) shape: {out.shape}")

        return out, reg_loss
