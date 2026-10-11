import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

text = open("/storage/slurm/home/68070701616@cpe.kmutt.ac.th/DeepLearning/src/Lab8/data/clean.txt", encoding="utf-8").read()

chars = sorted(set(text)) # separate each char
stoi = {ch: i for i, ch in enumerate(chars)} # 'a': 0, char to int (mapping)
itos = {i: ch for ch, i in stoi.items()} # 0 : 'a' , int to char (reverse mapping)
vocab_size = len(chars) # char size

data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long) # to tensor

def batchify(data, batch_size):
    n = data.size(0) // batch_size  # n char per batch = char / batch_size
    data = data[: n * batch_size] # cutting
    return data.view(batch_size, n) # to list form (batch_size row with n elements per row)

def get_batches(source, seq_len):
    for i in range(0, source.size(1) - 1, seq_len): # From 0 to n with seq_len steps
        # print(i)
        length = min(seq_len, source.size(1) - 1 - i) # get length (noemallu is seq_size but sometime last round maybe smaller than seq_len)
        x = source[:, i : i + length] # from start to stop
        y = source[:, i + 1 : i + 1 + length] # next element of x sequence
        yield x, y

# Train Test Split
split = int(len(data) * 0.9)
train_data = batchify(data[:split], batch_size=32)
val_data = batchify(data[split:], batch_size=32)

# ---------------------------- Model Section ---------------------------------

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.lstm1 = nn.LSTM(embed_dim, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.fc.weight = self.embedding.weight

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = [None, None]

        h = self.drop(self.embedding(x))
        h, h1 = self.lstm1(h, hidden[0])
        h = self.drop(h)
        h, h2 = self.lstm2(h, hidden[1])
        h = self.drop(h)

        return self.fc(h), [h1, h2]


def detach_hidden(hidden):
    return [(h.detach(), c.detach()) for h, c in hidden]

@torch.no_grad() #not train me pleaseeeee
def evaluate(model, data, seq_len=128):
    model.eval()
    total, count, hidden = 0.0, 0, None
    for x, y in get_batches(data, seq_len):
        x, y = x.to(device), y.to(device)
        logits, hidden = model(x, hidden)
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
        total += loss.item() * y.numel()
        count += y.numel()
    return total / count

model = LSTMLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
loss_fn = nn.CrossEntropyLoss()
best_val = float("inf")

# Parameter Setting
seq_len = 128
epochs = 40

train_loss_history = []
val_loss_history = []
ppl_loss_history = []

for epoch in range(epochs):
    model.train()
    hidden, total, steps = None, 0.0, 0
    for x, y in get_batches(train_data, seq_len):
        x, y = x.to(device), y.to(device)
        if hidden is not None:
            hidden = detach_hidden(hidden)

        logits, hidden = model(x, hidden)
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total += loss.item()
        steps += 1

    train_loss = total / steps
    train_loss_history.append(train_loss)

    val_loss = evaluate(model, val_data, seq_len)
    val_loss_history.append(val_loss)
    ppl_loss_history.append(math.exp(val_loss))

    print(f"epoch {epoch:2d}  train {train_loss:.3f}  val {val_loss:.3f} ppl {math.exp(val_loss):.2f}")

    # Best pleaseeeee
    if val_loss < best_val:
        best_val = val_loss
        torch.save({"model_state": model.state_dict(), "vocab_size": vocab_size,
                    "stoi": stoi, "itos": itos}, "lstm_lm_best.pt")

# ----------------------------------- PLOT ------------------

def plot_single(data, data_lab, xlab, ylab, tit, save_name, save_dir="/storage/slurm/home/68070701616@cpe.kmutt.ac.th/DeepLearning/src/Lab8/Results/"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(len(data)), data, lw=2, label=data_lab)
    ax.set(xlabel=xlab, ylabel=ylab, title=tit)
    fig.savefig(save_dir + save_name)
    # plt.show()

def plot_compare(data_1, data_1_lab, data_2, data_2_lab, xlab, ylab, tit, save_name, save_dir="/storage/slurm/home/68070701616@cpe.kmutt.ac.th/DeepLearning/src/Lab8/Results/"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(len(data_1)), data_1, lw=2, label=data_1_lab)
    ax.plot(np.arange(len(data_2)), data_2, lw=2, label=data_2_lab)
    ax.set(xlabel=xlab, ylabel=ylab, title=tit)
    # ax.set_ylim(top=1)
    # ax.set_xlim(xmax=2000, xmin=0)
    plt.legend()
    # plt.show()
    fig.savefig(save_dir + save_name)

plot_single(train_loss_history, "Training loss", "Epochs", "Loss", "Training Loss", "t.png")
plot_single(val_loss_history, "Validation loss", "Epochs", "Loss", "Validation Loss", "v.png")
plot_single(ppl_loss_history, "Perplexity", "Epochs", "ppl", "Perplexity", "p.png")
plot_compare(train_loss_history, "Training loss", val_loss_history, "Validation loss", "Epochs", "Loss", "Train and Validation Loss Comparison", "cp.png")

# ----------------------------------- Generate Section -------------------------------------

def text_to_ids(text):
    ids = [stoi[ch] for ch in text if ch in stoi]
    if not ids:
        raise ValueError("Char not in training data")
    return ids

def to_tensor(ids):
    return torch.tensor([ids], device=device)

def pick_next_char(scores, temperature, top_k):
    scores = scores / temperature

    if top_k:
        cutoff = torch.topk(scores, top_k).values[-1]
        scores = scores.masked_fill(scores < cutoff, -float("inf"))

    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, 1).item()

@torch.no_grad()
def generate(model, prompt, length=500, temperature=0.8, top_k=20):
    model.eval()

    prompt_ids = text_to_ids(prompt)

    scores, memory = model(to_tensor(prompt_ids))
    last_scores = scores[0, -1]

    result = prompt
    for _ in range(length):
        next_id = pick_next_char(last_scores, temperature, top_k)
        result += itos[next_id]

        scores, memory = model(to_tensor([next_id]), memory)
        last_scores = scores[0, -1]

    return result

# load check point and to it.
ckpt = torch.load("lstm_lm_best.pt", map_location=device)

stoi, itos = ckpt["stoi"], ckpt["itos"]
model = LSTMLanguageModel(ckpt["vocab_size"]).to(device)
model.load_state_dict(ckpt["model_state"])

print(generate(model, "My father was a man of", length=400, temperature=0.8))