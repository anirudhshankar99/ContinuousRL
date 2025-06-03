import torch
import torch.nn as nn
import numpy as np

class Agent(nn.Module):
    def __init__(self, input_dim, output_dim, condition_dim, hidden_dim, n_heads, num_layers, seq_len):
        super().__init__()
        self.model = OrbitPredictorTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            num_layers=num_layers,
            seq_len=seq_len,
        )
        self.out_shape = input_dim // 2
    def forward(self, states, accs, conditioning_token):
        X, value = self.model(states, accs, conditioning_token)
        (means, log_stds) = torch.split(X, [self.out_shape, self.out_shape], dim=-1)
        return means, log_stds.exp(), value.sum()

class OrbitPredictorTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, condition_dim, hidden_dim, n_heads, num_layers, seq_len):
        super().__init__()
        self.seq_len = seq_len
        acc_dim = input_dim // 2

        # Input Embeddings
        self.state_embedding = nn.Linear(input_dim, hidden_dim)
        self.acc_embedding = nn.Linear(acc_dim, hidden_dim)
        self.conditioning_embedding = nn.Linear(condition_dim, hidden_dim)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len + 1, hidden_dim))  # +1 for conditioning token

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction Head (predicts future state)
        self.actor_head = nn.Linear(hidden_dim, output_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, states, accs, conditioning_token):
        # states: (T, input_dim)
        # accs:   (T, acc_dim)
        # conditioning_token: (cond_dim)

        T, _ = states.shape

        # Embed input
        embedded_state = self.state_embedding(states)
        embedded_acc = self.acc_embedding(accs)
        embedded_tokens = embedded_state + embedded_acc

        # Conditioning token
        conditioning_token = self.conditioning_embedding(conditioning_token).unsqueeze(0)  # (1, hidden_dim)

        # Concatenate conditioning_token + sequence
        x = torch.cat([conditioning_token, embedded_tokens], dim=0)  # (T+1, hidden_dim)

        # Add positional encoding
        x = x + self.positional_encoding[:, :T+1, :]

        # Create causal mask (shape: T+1 x T+1)
        attention_mask = torch.triu(torch.ones(T+1, T+1), diagonal=1).bool().to(x.device)

        # Transformer encoding
        x = self.transformer(x, mask=attention_mask)

        # Discard conditioning token before prediction
        x = x[:, 1:, :]  # (T, hidden_dim)

        # Predict next states
        predictions = self.actor_head(x)  # (T, input_dim)
        value = self.critic_head(x)
        return predictions.squeeze(0), value.squeeze(0)