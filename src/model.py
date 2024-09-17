import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
from utils import MultiHeadAttention

class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.fusion_dim,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout_rate,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.attention_pool = nn.MultiheadAttention(config.fusion_dim, config.num_heads, batch_first=True)

        self.projection_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.fusion_dim, config.fusion_dim),
                nn.LayerNorm(config.fusion_dim),
                nn.GELU(),
                nn.Dropout(config.dropout_rate)
            ) for _ in range(4)
        ])

        self.latent_projection = nn.Linear(config.fusion_dim, config.latent_dim)
        self.final_layer_norm = nn.LayerNorm(config.latent_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.encoder(x)
        attn_output, attn_weights = self.attention_pool(x, x, x)
        weighted_sum = x + torch.bmm(attn_weights, x)

        for layer in self.projection_layers:
            weighted_sum = layer(weighted_sum) + weighted_sum  # Residual connection

        pooled = weighted_sum.mean(dim=1)
        latent = self.latent_projection(pooled)
        latent = self.final_layer_norm(self.dropout(F.gelu(latent)))
        return latent

class SiameseDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.latent_dim,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout_rate,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=config.num_decoder_layers)

        self.output_projection = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.latent_dim, config.vocab_size)
        )

        self.esm_projection = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.latent_dim, 1280)
        )

        esm_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        self.lm_head = esm_model.lm_head

    def forward(self, latent, use_lm_head=True):
        batch_size = latent.size(0)
        seq_len = self.config.max_protein_length

        decoded = self.decoder(
            tgt=torch.zeros(batch_size, seq_len, self.config.latent_dim).to(latent.device),
            memory=latent.unsqueeze(1).repeat(1, seq_len, 1)
        )

        if use_lm_head:
            decoded_projected = self.esm_projection(decoded)
            logits = self.lm_head(decoded_projected)
        else:
            logits = self.output_projection(decoded)

        return logits

class CombinedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rna_projection = self.create_projection_layers(config.rna_embed_dim, config.fusion_dim)
        self.protein_projection = self.create_projection_layers(config.protein_embed_dim, config.fusion_dim)
        self.siamese_network = SiameseNetwork(config)
        self.siamese_decoder = SiameseDecoder(config)
        esm_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        self.lm_head = esm_model.lm_head

    def create_projection_layers(self, input_dim, output_dim):
        layers = []
        current_dim = input_dim
        for _ in range(3):  # 3 intermediate layers
            layers.extend([
                nn.Linear(current_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout_rate)
            ])
            current_dim = output_dim
        layers.append(nn.Linear(current_dim, output_dim))  # Final projection
        return nn.Sequential(*layers)

    def forward(self, rna_emb, protein_emb=None, use_lm_head=True):
        rna_proj = self.rna_projection(rna_emb)
        rna_latent = self.siamese_network(rna_proj)

        if protein_emb is not None:
            protein_proj = self.protein_projection(protein_emb)
            protein_latent = self.siamese_network(protein_proj)
        else:
            protein_latent = None

        generated_sequence = self.siamese_decoder(rna_latent, use_lm_head)

        return rna_latent, protein_latent, generated_sequence
