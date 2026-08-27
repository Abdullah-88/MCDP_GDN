import torch
from torch import nn

def l2norm(x, dim = -1, eps = 1e-6):
    return x * torch.rsqrt((x * x).sum(dim = dim, keepdim = True) + eps)

class GatedDeltaNet(nn.Module):
    def __init__(
        self, d_in, d_out, dropout, num_heads, qkv_bias = False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
     
        self.W_gate = nn.Linear(d_in, d_out, bias = False)
        self.W_beta = nn.Linear(d_in, d_out, bias = False)

        self.W_alpha = nn.Linear(d_in, num_heads, bias = False)
        self.dt_bias = nn.Parameter(torch.ones(num_heads))
        A_init = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A_init))
       
        self.norm = nn.RMSNorm(self.head_dim, eps = 1e-6)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
       
        beta = torch.sigmoid(self.W_beta(x))
        alpha_log = -self.A_log.exp().view(1, 1, -1) * F.softplus(
            self.W_alpha(x) + self.dt_bias
        )
        alpha = alpha_log.exp()
        gate = self.W_gate(x)
       
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        beta = beta.view(b, num_tokens, self.num_heads, self.head_dim)
        gate = gate.view(b, num_tokens, self.num_heads, self.head_dim)  

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        beta = beta.transpose(1, 2)

        queries = l2norm(queries, dim = -1) / (self.head_dim ** 0.5)
        keys = l2norm(keys, dim = -1)
       
        S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)

        outs = []
       
        for t in range(num_tokens):
            k_t = keys[:, :, t]
            q_t = queries[:, :, t]
            v_t = values[:, :, t]
            b_t = beta[:, :, t]
            a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

            S = S * a_t
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * b_t
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
          
            outs.append(y_t)

        context = torch.stack(outs, dim=2).transpose(1, 2).contiguous()
        context = context.view(b, num_tokens, self.num_heads, self.head_dim)

        context = self.norm(context)
        context = context * F.silu(gate)
      
        context = context.view(b, num_tokens, self.d_out)
        context = self.dropout(context)
        out = self.out_proj(context)
        return out
        
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MCGatingUnit(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        
        self.gdn_1 = GatedDeltaNet(dim, dim, dropout, 8)     
        self.gdn_2 = GatedDeltaNet(dim, dim, dropout, 8)    
       
    def forward(self, x):
        u, v = x, x 
        u = self.gdn_1(u)   
        v = self.gdn_1(v)
        out = u * v
        return out

class MCDPGDNBlock(nn.Module):
    def __init__(self, d_model, d_ffn, dropout):
        super().__init__()
       
        self.norm = nn.LayerNorm(d_model)       
        self.mcgu = MCGatingUnit(d_model, dropout)
        self.ffn = FeedForward(d_model, d_ffn, dropout)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mcgu(x)   
        x = x + residual      
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        out = x + residual
        return out

class MCDPGDN(nn.Module):
    def __init__(self, d_model, d_ffn, num_layers, dropout):
        super().__init__()
        
        self.model = nn.Sequential(
            *[MCDPGDNBlock(d_model, d_ffn, dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)
