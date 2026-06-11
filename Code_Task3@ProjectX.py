import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip().split() for line in f.readlines()]

inputs = load_data(r'C:\\Users\\luvja\\Desktop\\PROJECT X SUBMISSION\\attention challenge\\dataset\\input_texts.txt')
labels = load_data(r'C:\\Users\\luvja\\Desktop\\PROJECT X SUBMISSION\\attention challenge\\dataset\\label_texts.txt')

vocab = {'<pad>': 0, '<unk>': 1}
for sentence in inputs + labels:
    for word in sentence:
        if word not in vocab:
            vocab[word] = len(vocab)

vocab_size = len(vocab)

def encode(sentences, vocab, max_len=20):
    encoded = []
    for seq in sentences:
        num_seq = [vocab.get(word, vocab['<unk>']) for word in seq]
        if len(num_seq) < max_len:
            num_seq += [vocab['<pad>']] * (max_len - len(num_seq))
        else:
            num_seq = num_seq[:max_len]
        encoded.append(num_seq)
    return torch.tensor(encoded)

MAX_LEN = 20
X = encode(inputs, vocab, MAX_LEN)
Y = encode(labels, vocab, MAX_LEN)

class DialogDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self): 
        return len(self.x)
    def __getitem__(self, idx): 
        return self.x[idx], self.y[idx]

dataloader = DataLoader(DialogDataset(X, Y), batch_size=32, shuffle=True)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        
        Q = self.q_linear(q).view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(k).view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(v).view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, V).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        
        return self.out(out)

class StudentTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.enc_attention = MultiHeadAttention(d_model, heads)
        self.enc_ff = nn.Sequential(nn.Linear(d_model, d_model*2), nn.ReLU(), nn.Linear(d_model*2, d_model))
        self.enc_norm1 = nn.LayerNorm(d_model)
        self.enc_norm2 = nn.LayerNorm(d_model)

        self.dec_attention1 = MultiHeadAttention(d_model, heads)
        self.dec_attention2 = MultiHeadAttention(d_model, heads)
        self.dec_ff = nn.Sequential(nn.Linear(d_model, d_model*2), nn.ReLU(), nn.Linear(d_model*2, d_model))
        self.dec_norm1 = nn.LayerNorm(d_model)
        self.dec_norm2 = nn.LayerNorm(d_model)
        self.dec_norm3 = nn.LayerNorm(d_model)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt):
        B, T_tgt = tgt.size()
        tgt_mask = torch.tril(torch.ones(T_tgt, T_tgt)).unsqueeze(0).unsqueeze(0).to(src.device)
        
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        
        enc_out = self.enc_norm1(src_emb + self.enc_attention(src_emb, src_emb, src_emb))
        enc_out = self.enc_norm2(enc_out + self.enc_ff(enc_out))
        
        dec_out = self.dec_norm1(tgt_emb + self.dec_attention1(tgt_emb, tgt_emb, tgt_emb, tgt_mask))
        dec_out = self.dec_norm2(dec_out + self.dec_attention2(dec_out, enc_out, enc_out))
        dec_out = self.dec_norm3(dec_out + self.dec_ff(dec_out))
        
        return self.fc_out(dec_out)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StudentTransformer(vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0) 
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_tokens = 0
    total_tokens = 0
    
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]
        
        optimizer.zero_grad()
        
        output = model(src, tgt_input)
        loss = criterion(output.reshape(-1, vocab_size), tgt_expected.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        predictions = output.argmax(dim=-1)
        mask = (tgt_expected != 0) 
        
        correct_tokens += ((predictions == tgt_expected) & mask).sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")