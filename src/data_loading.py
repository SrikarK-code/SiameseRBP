import torch
from torch.utils.data import Dataset, DataLoader
import random

class RBPDataset(Dataset):
    def __init__(self, data, config):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        rna_emb = torch.tensor(item['rna_motif_emb'], dtype=torch.float32)
        protein_emb = torch.tensor(item['rbp_esm_emb'], dtype=torch.float32)
        protein_seq = torch.tensor([self.config.aa_to_idx[aa] for aa in item['RBP_sequence']], dtype=torch.long)
        return rna_emb, protein_emb, protein_seq

def collate_fn(batch, config):
    rna_embs, protein_embs, protein_seqs = zip(*batch)

    max_rna_len = max(emb.size(0) for emb in rna_embs)
    max_protein_len = max(emb.size(0) for emb in protein_embs)

    padded_rna_emb = torch.zeros(len(batch), max_rna_len, config.rna_embed_dim)
    padded_protein_emb = torch.zeros(len(batch), max_protein_len, config.protein_embed_dim)
    padded_protein_seq = torch.full((len(batch), max_protein_len), config.pad_token)

    for i, (rna_emb, protein_emb, protein_seq) in enumerate(zip(rna_embs, protein_embs, protein_seqs)):
        padded_rna_emb[i, :rna_emb.size(0)] = rna_emb
        padded_protein_emb[i, :protein_emb.size(0)] = protein_emb
        padded_protein_seq[i, :protein_seq.size(0)] = protein_seq

    return padded_rna_emb, padded_protein_emb, padded_protein_seq

def get_dataloader(dataset, config):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, 
                      collate_fn=lambda x: collate_fn(x, config))
