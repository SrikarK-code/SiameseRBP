from data_process import load_and_process_data
from data import RBPDataset, get_dataloader
from model import CombinedModel
from train import train_and_validate
import torch
from sklearn.model_selection import train_test_split
import wandb

class Config:
    vocab_size = 33  # ESM-2 vocabulary size
    pad_token = 0
    start_token = 1
    end_token = 2
    rna_embed_dim = 120
    protein_embed_dim = 1280
    fusion_dim = 512
    latent_dim = 256
    d_model = 512
    num_heads = 8
    num_layers = 6
    num_decoder_layers = 6
    d_ff = 2048
    dropout_rate = 0.1
    max_rna_length = 100
    max_protein_length = 200
    batch_size = 32
    num_epochs = 10
    learning_rate = 3e-4
    aa_to_idx = {aa: idx for idx, aa in enumerate("ARNDCEQGHILKMFPSTWYV")}
    val_split = 0.2

def main():
    config = Config()
    
    # Load and process data
    data = load_and_process_data('final_attract_db_with_emb.csv', 'rbp_seqs_dict.pkl', 'rna_motif_emb.npy')
    
    # Split data into train and validation sets
    train_data, val_data = train_test_split(data, test_size=config.val_split, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = RBPDataset(train_data, config)
    val_dataset = RBPDataset(val_data, config)
    train_dataloader = get_dataloader(train_dataset, config, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, config, shuffle=False)
    
    # Initialize model
    model = CombinedModel(config)
    
    # Train and validate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb
    wandb.init(project="rbp_predictor", config=vars(config))
    
    trained_model = train_and_validate(model, train_dataloader, val_dataloader, config, device)
    
    # Save the trained model
    torch.save(trained_model.state_dict(), 'trained_rbp_model.pth')
    print("Training completed and model saved.")
    
    wandb.finish()

if __name__ == "__main__":
    main()
