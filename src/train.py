import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(model, train_loader, config, device):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    contrastive_loss = nn.CosineEmbeddingLoss()
    ce_loss = nn.CrossEntropyLoss(ignore_index=config.pad_token)

    model.to(device)
    model.train()

    for epoch in range(config.num_epochs):
        total_loss = 0
        for rna_embs, protein_embs, target_seqs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            rna_embs, protein_embs, target_seqs = rna_embs.to(device), protein_embs.to(device), target_seqs.to(device)

            optimizer.zero_grad()

            rna_latent, protein_latent, generated_sequence = model(rna_embs, protein_embs, use_lm_head=True)

            # Assuming all pairs are positive for this example
            labels = torch.ones(rna_latent.size(0)).to(device)
            loss_contrastive = contrastive_loss(rna_latent, protein_latent, labels)
            loss_generation = ce_loss(generated_sequence.view(-1, config.vocab_size), target_seqs.view(-1))

            loss = loss_contrastive + loss_generation
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {avg_loss:.4f}")

    return model
