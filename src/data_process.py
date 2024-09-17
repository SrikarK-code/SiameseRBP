import pandas as pd
import numpy as np
import pickle

def load_and_process_data(csv_file, pkl_file, npy_file):
    # Load CSV data
    data = pd.read_csv(csv_file)
    
    # Load RBP sequences dictionary
    with open(pkl_file, "rb") as f:
        rbp_seqs_dict = pickle.load(f)
    
    # Load RNA motif embeddings
    rna_motif_emb = np.load(npy_file, allow_pickle=True)
    
    # Process data
    data = data.drop(columns=['rna_motif_emb', 'rbp_esm_emb'])
    data['rna_motif_emb'] = rna_motif_emb
    data['rbp_esm_emb'] = data['RBP_sequence'].map(rbp_seqs_dict)
    
    # Convert list of tensors to numpy array
    data['rbp_esm_emb'] = data['rbp_esm_emb'].apply(tensors_to_numpy)
    
    return data

def tensors_to_numpy(tensor_list):
    return np.stack([t.numpy() for t in tensor_list])
