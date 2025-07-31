import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        B, Q_len, _ = query.shape
        K_len = key.shape[1]

        q = self.q_proj(query).reshape(B, Q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, K_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, K_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, Q_len, self.embed_dim)
        out = self.out_proj(attn_output)

        return out

class SAM(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(SAM, self).__init__()
        self.embed_dim = embed_dim

        self.cross_attn = CrossAttention(embed_dim=embed_dim, num_heads=1)
        self.linear_proj = nn.Linear(embed_dim, embed_dim)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.eye_(m.weight) if m.weight.shape[0] == m.weight.shape[1] else nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(self, q_feat, k_feat, v_feat):
        q_feat = self.ln1(q_feat)
        k_feat = self.ln2(k_feat)
        v_feat = self.ln3(v_feat)

        attn_out = self.cross_attn(q_feat, k_feat, v_feat)
        attn_out = self.ln(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.ln(out)

        return out

if __name__ == "__main__":
    model = SAM()
    b, c, f, d = 32, 6, 12, 512
    c_feat = torch.rand(b, c, d)
    f_feat = torch.rand(b, f, d)
    out = model(c_feat, f_feat, f_feat)
    out = model(f_feat, c_feat, c_feat)
