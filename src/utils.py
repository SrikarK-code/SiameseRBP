import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.d_k = config.d_model // config.num_heads

        self.q_linear = nn.Linear(config.d_model, config.d_model)
        self.k_linear = nn.Linear(config.d_model, config.d_model)
        self.v_linear = nn.Linear(config.d_model, config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

    def forward(self, x):
        bs = x.size(0)
        q = self.q_linear(x).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(context)
        return output

def generate_peptides(model, token_representations, num_samples, sample_variances, config):
    generated_peptides = []
    aa_toks = list("ARNDCEQGHILKMFPSTWYV")
    aa_idxs = [config.aa_to_idx[aa] for aa in aa_toks]

    for i in sample_variances:
        for j in range(num_samples):
            gen_pep = token_representations + torch.randn(token_representations.shape) * i * token_representations.var()
            aa_logits = model.lm_head(gen_pep.cuda())[:, :, aa_idxs]
            predictions = torch.argmax(aa_logits, dim=2).tolist()[0]
            generated_pep_seq = "".join([aa_toks[i] for i in predictions])
            generated_peptides.append(generated_pep_seq[1:-1])

    return generated_peptides
