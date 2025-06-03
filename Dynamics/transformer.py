import torch
import torch.nn as nn
import numpy as np

# Actor module
class Actor(nn.Module):
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
        self.out_shape = output_dim
    def forward(self, states, accs, conditioning_token):
        X = self.model(states, accs, conditioning_token)
        (means, log_stds) = torch.split(X, [self.out_shape, self.out_shape], dim=-1)
        return means, log_stds.exp()
    
# Critic module
class Critic(nn.Module):
    def __init__(self, env, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 64),
            activation(),
            nn.Linear(64, 32),
            activation(),
            nn.Linear(32, 1)
        )
    
    def forward(self, X):
        return self.model(X)

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
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, n_heads=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction Head (predicts future state)
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, states, accs, conditioning_token):
        # states: (B, T, input_dim)
        # accs:   (B, T, acc_dim)
        # conditioning_token: (B, cond_dim)

        B, T, _ = states.shape

        # Embed input
        embedded_state = self.state_embedding(states)
        embedded_acc = self.acc_embedding(accs)
        embedded_tokens = embedded_state + embedded_acc

        # Conditioning token
        conditioning_token = self.conditioning_embedding(conditioning_token).unsqueeze(1)  # (B, 1, hidden_dim)

        # Concatenate conditioning_token + sequence
        x = torch.cat([conditioning_token, embedded_tokens], dim=1)  # (B, T+1, hidden_dim)

        # Add positional encoding
        x = x + self.positional_encoding[:, :T+1, :]

        # Create causal mask (shape: T+1 x T+1)
        attention_mask = torch.triu(torch.ones(T+1, T+1), diagonal=1).bool().to(x.device)

        # Transformer encoding
        x = self.transformer(x, mask=attention_mask)

        # Discard conditioning token before prediction
        x = x[:, 1:, :]  # (B, T, hidden_dim)

        # Predict next states
        predictions = self.output_head(x)  # (B, T, input_dim)

        return predictions