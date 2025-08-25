import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicLocalGlobalRouter(nn.Module):
    def __init__(self, seq_len, num_heads, local_radius=4, head_dim=64):
        super().__init__()
        self.local_radius = local_radius
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sigmoid_gate = nn.Sequential(
            nn.Linear(head_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, q, k, v, global_out, buckets):
        print(f"[DynamicLocalGlobalRouter] q shape {q.shape}, global_out shape {global_out.shape}")
        batch, heads, seq_len, head_dim = q.size()
        local_out = torch.zeros_like(global_out)
        for h in range(heads):
            for t in range(seq_len):
                l_start = max(0, t - self.local_radius)
                l_end = min(seq_len, t + self.local_radius + 1)
                q_t = q[:,h,t].unsqueeze(1)
                k_local = k[:,h,l_start:l_end]
                v_local = v[:,h,l_start:l_end]
                dots = torch.matmul(q_t, k_local.transpose(1,2)) / (head_dim ** 0.5)
                attn = torch.softmax(dots, dim=-1)
                local = torch.matmul(attn, v_local)
                local_out[:,h,t] = local.squeeze(1)
        print(f"  local_out shape: {local_out.shape}")

        gate_input = q.mean(dim=2)
        gate = self.sigmoid_gate(gate_input)
        print(f"  gate shape before expand: {gate.shape}")
        gate = gate.expand(-1, -1, seq_len)
        gate = gate.unsqueeze(-1)
        print(f"  gate shape after expand: {gate.shape}")

        mixed = gate * local_out + (1 - gate) * global_out
        print(f"  mixed (output) shape: {mixed.shape}")
        return mixed
