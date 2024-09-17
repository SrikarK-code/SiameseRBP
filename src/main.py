from data_process import load_and_process_data
from data import RBPDataset, get_dataloader
from model import CombinedModel
from train import train
import torch

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

def main():
    config = Config()

    # Load and process data
    data = load_and_process_data('final_attract_db_with_emb.csv', 'rbp_seqs_dict.pkl', 'rna_motif_emb.npy')

    # Create dataset and dataloader
    dataset = RBPDataset(data, config)
    dataloader = get_dataloader(dataset, config)

    # Initialize model
    model = CombinedModel(config)

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train(model, dataloader, config, device)

    # Save the trained model
    torch.save(trained_model.state_dict(), 'trained_rbp_model.pth')

    print("Training completed and model saved.")

if __name__ == "__main__":
    main()
