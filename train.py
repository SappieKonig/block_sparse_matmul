import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math
from tqdm import tqdm


class AutoregressiveTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, d_ff=1024, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_ff,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        x = self.embedding(x) + self.pos_embedding(pos)
        
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        x = self.transformer(x, mask=mask)
        
        return self.lm_head(x)


class Enwik9Dataset(Dataset):
    def __init__(self, file_path, seq_len=512):
        self.seq_len = seq_len
        
        with open(file_path, 'rb') as f:
            data = f.read()
        
        self.tokens = list(data)
        self.num_sequences = len(self.tokens) // seq_len
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        sequence = torch.tensor(self.tokens[start:start + self.seq_len], dtype=torch.long)
        return sequence[:-1], sequence[1:]


def main():
    # Model configuration
    vocab_size = 256
    d_model = 256
    nhead = 8
    num_layers = 6
    d_ff = 1024
    seq_len = 512
    batch_size = 128
    learning_rate = 1e-4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    dataset = Enwik9Dataset('data/enwik9', seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = AutoregressiveTransformer(
        vocab_size=vocab_size, 
        d_model=d_model, 
        nhead=nhead, 
        num_layers=num_layers, 
        d_ff=d_ff, 
        max_len=seq_len
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    embedding_params = model.embedding.weight.numel() + model.pos_embedding.weight.numel()
    non_embedding_params = total_params - embedding_params
    
    print(f'Total parameters: {total_params:,}')
    print(f'Non-embedding parameters: {non_embedding_params:,}')
    print(f'Training on {len(dataset)} sequences')
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    model.train()
    losses = []
    
    for epoch in range(1):
        for batch_idx, (input_seq, target_seq) in enumerate(tqdm(dataloader)):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            
            optimizer.zero_grad()
            logits = model(input_seq)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_seq.reshape(-1))
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_loss = sum(losses[-100:]) / min(len(losses), 100)
                print(f'Batch {batch_idx}, Avg Loss (last 100): {avg_loss:.4f}')
    
    torch.save(model, 'transformer_model.pth')
    print('Model saved to transformer_model.pth')


if __name__ == '__main__':
    main()