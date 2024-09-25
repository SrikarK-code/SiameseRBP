import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

def train_and_validate(model, train_loader, val_loader, config, device):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    contrastive_loss = nn.CosineEmbeddingLoss()
    ce_loss = nn.CrossEntropyLoss(ignore_index=config.pad_token)
    model.to(device)
    
    wandb.init(project="rbp_predictor", config=vars(config))
    
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        total_train_loss = 0
        train_contrastive_loss = 0
        train_generation_loss = 0
        
        for rna_embs, protein_embs, target_seqs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} - Train"):
            rna_embs, protein_embs, target_seqs = rna_embs.to(device), protein_embs.to(device), target_seqs.to(device)
            optimizer.zero_grad()
            rna_latent, protein_latent, generated_sequence = model(rna_embs, protein_embs, use_lm_head=True)
            
            labels = torch.ones(rna_latent.size(0)).to(device)
            loss_contrastive = contrastive_loss(rna_latent, protein_latent, labels)
            loss_generation = ce_loss(generated_sequence.view(-1, config.vocab_size), target_seqs.view(-1))
            loss = loss_contrastive + loss_generation
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            train_contrastive_loss += loss_contrastive.item()
            train_generation_loss += loss_generation.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_contrastive_loss = train_contrastive_loss / len(train_loader)
        avg_train_generation_loss = train_generation_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_contrastive_loss = 0
        val_generation_loss = 0
        
        with torch.no_grad():
            for rna_embs, protein_embs, target_seqs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} - Validation"):
                rna_embs, protein_embs, target_seqs = rna_embs.to(device), protein_embs.to(device), target_seqs.to(device)
                rna_latent, protein_latent, generated_sequence = model(rna_embs, protein_embs, use_lm_head=True)
                
                labels = torch.ones(rna_latent.size(0)).to(device)
                loss_contrastive = contrastive_loss(rna_latent, protein_latent, labels)
                loss_generation = ce_loss(generated_sequence.view(-1, config.vocab_size), target_seqs.view(-1))
                loss = loss_contrastive + loss_generation
                
                total_val_loss += loss.item()
                val_contrastive_loss += loss_contrastive.item()
                val_generation_loss += loss_generation.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_contrastive_loss = val_contrastive_loss / len(val_loader)
        avg_val_generation_loss = val_generation_loss / len(val_loader)
        
        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_contrastive_loss": avg_train_contrastive_loss,
            "train_generation_loss": avg_train_generation_loss,
            "val_loss": avg_val_loss,
            "val_contrastive_loss": avg_val_contrastive_loss,
            "val_generation_loss": avg_val_generation_loss
        })
        
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_rbp_model.pth')
            print("New best model saved!")
    
    wandb.finish()
    return model
