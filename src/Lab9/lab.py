import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')

from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig

# for reproducibility
# refer https://pytorch.org/docs/stable/notes/randomness.html
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# # plot style configuration
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

rcParams['figure.figsize'] = 12, 8

#-------------------------- Load Data

base = "hf://datasets/abisee/cnn_dailymail/1.0.0"

# import file from all training file and concat datasets
train_df = pd.concat(
    [pd.read_parquet(f"{base}/train-{i:05d}-of-00003.parquet") for i in range(1)],
    ignore_index=True,
)
valid_df   = pd.read_parquet(f"{base}/validation-00000-of-00001.parquet")
test_df  = pd.read_parquet(f"{base}/test-00000-of-00001.parquet")

for df in (train_df, valid_df, test_df):
    df.rename(columns={"article": "source", "highlights": "target"}, inplace=True)
    df.drop(columns=["id"], inplace=True)

print(train_df.shape, valid_df.shape, test_df.shape)

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize_fn(examples):
    model_inputs = tokenizer(examples['source'], truncation=True, max_length=128)
    labels = tokenizer(examples['target'], truncation=True, max_length=128)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(tokenize_fn, batched=True)
valid_dataset = valid_dataset.map(tokenize_fn, batched=True)
test_dataset = test_dataset.map(tokenize_fn, batched=True)

# Remove the raw string columns
cols_to_remove = ["source", "target"]
train_dataset = train_dataset.remove_columns(cols_to_remove)
valid_dataset = valid_dataset.remove_columns(cols_to_remove)
test_dataset = test_dataset.remove_columns(cols_to_remove)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, 
    label_pad_token_id=tokenizer.pad_token_id, 
    return_tensors="pt"
)

BATCH_SIZE = 8

train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
valid_iterator = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)

PAD_IDX = tokenizer.pad_token_id
SOS_IDX = tokenizer.cls_token_id # BERT uses [CLS] as sequence start
EOS_IDX = tokenizer.sep_token_id # BERT uses [SEP] as sequence end
UNK_IDX = tokenizer.unk_token_id

#------------------------- Model

class ScaledDotProductAttention(nn.Module):
    """ Computes scaled dot product attention
    """

    def __init__(self, scale, dropout_rate):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, query, key, value, mask=None):
        """ query: (batch_size, n_heads, query_len, head_dim)
            key: (batch_size, n_heads, key_len, head_dim)
            value: (batch_size, n_heads, value_len, head_dim)
            mask: (batch_size, 1, 1, source_seq_len) for source mask
                  (batch_size, 1, target_seq_len, target_seq_len) for target mask
        """
        # calculate alignment scores
        scores = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, n_heads, query_len, value_len)
        scores = scores / self.scale  # (batch_size, num_heads, query_len, value_len)

        # mask out invalid positions
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  # (batch_size, n_heads, query_len, value_len)

        # calculate the attention weights (prob) from alignment scores
        attn_probs = F.softmax(scores, dim=-1)  # (batch_size, n_heads, query_len, value_len)

        # calculate context vector
        output = torch.matmul(self.dropout(attn_probs), value)  # (batch_size, n_heads, query_len, head_dim)

        # output: (batch_size, n_heads, query_len, head_dim)
        # attn_probs: (batch_size, n_heads, query_len, value_len)
        return output, attn_probs

class MultiHeadAttention(nn.Module):
    """ Implements Multi-Head Self-Attention proposed by Vaswani et al., 2017.
        refer https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model, n_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "`d_model` should be a multiple of `n_heads`"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads  # head_dim
        self.dropout_rate = dropout_rate

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(np.sqrt(self.d_k), dropout_rate)


    def split_heads(self, x):
        """ x: (batch_size, seq_len, d_model)
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)

        # x: (batch_size, n_heads, seq_len, head_dim)
        return x


    def group_heads(self, x):
        """ x: (batch_size, n_heads, seq_len, head_dim)
        """
        batch_size = x.size(0)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        # x: (batch_size, seq_len, d_model)
        return x


    def forward(self, query, key, value, mask=None):
        """ query: (batch_size, query_len, d_model)
            key: (batch_size, key_len, d_model)
            value: (batch_size, value_len, d_model)
            mask: (batch_size, 1, source_seq_len) for source mask
                  (batch_size, target_seq_len, target_seq_len) for target mask
        """
        # apply linear projections to query, key and value
        Q = self.split_heads(self.W_q(query))  # (batch_size, n_heads, query_len, head_dim)
        K = self.split_heads(self.W_k(key))  # (batch_size, n_heads, key_len, head_dim)
        V = self.split_heads(self.W_v(value))  # (batch_size, n_heads, value_len, head_dim)

        if mask is not None:
            # apply same mask for all the heads
            mask = mask.unsqueeze(1)

            # mask: (batch_size, 1, 1, source_seq_len) for source mask
            #       (batch_size, 1, target_seq_len, target_seq_len) for target mask

        # calculate attention weights and context vector for each of the heads
        x, attn = self.attention(Q, K, V, mask)

        # x: (batch_size, n_heads, query_len, head_dim)
        # attn: (batch_size, n_heads, query_len, value_len)

        # concatenate context vector of all the heads
        x = self.group_heads(x)  # (batch_size, query_len, d_model)

        # apply linear projection to concatenated context vector
        x = self.W_o(x)  # (batch_size, query_len, d_model)

        # x: (batch_size, query_len, d_model)
        # attn: (batch_size, n_heads, query_len, value_len)
        return x, attn

class PositionwiseFeedForward(nn.Module):
    """ Implements a two layer feed-forward network.
    """

    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        """ x: (batch_size, seq_len, d_model)
        """
        x = self.dropout(F.relu(self.w_1(x)))  # (batch_size, seq_len, d_ff)
        x = self.w_2(x)  # (batch_size, seq_len, d_model)

        # x: (batch_size, seq_len, d_model)
        return x

class PositionalEncoding(nn.Module):
    """ Implements the sinusoidal positional encoding.
    """

    def __init__(self, d_model, dropout_rate=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        self.dropout = nn.Dropout(dropout_rate)

        # compute positional encodings
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:x.size(0), :]  # (batch_size, seq_len, d_model)
        x = self.dropout(x)  # (batch_size, seq_len, d_model)

        # x: (batch_size, seq_len, d_model)
        return x

class EncoderLayer(nn.Module):
    """ Encoder is made up of a self-attention layer and a feed-forward layer.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

        self.attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.ff_layer = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.ff_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x, mask):
        """ x: (batch_size, source_seq_len, d_model)
            mask: (batch_size, 1, source_seq_len)
        """
        # apply self-attention
        x1, _ = self.attn_layer(x, x, x, mask)  # (batch_size, source_seq_len, d_model)

        # apply residual connection followed by layer normalization
        x = self.attn_layer_norm(x + self.dropout(x1))  # (batch_size, source_seq_len, d_model)

        # apply position-wise feed-forward
        x1 = self.ff_layer(x)  # (batch_size, source_seq_len, d_model)

        # apply residual connection followed by layer normalization
        x = self.ff_layer_norm(x + self.dropout(x1))  # (batch_size, source_seq_len, d_model)

        # x: (batch_size, source_seq_len, d_model)
        return x

class Encoder(nn.Module):
    """ Encoder block is a stack of N identical encoder layers.
    """

    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, pad_idx, dropout_rate=0.1, max_len=5000):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.pad_idx = pad_idx
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = PositionalEncoding(d_model, dropout_rate, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, x, mask):
        """ x: (batch_size, source_seq_len)
            mask: (batch_size, 1, source_seq_len)
        """
        # apply positional encoding to token sequences
        x = self.tok_embedding(x)  # (batch_size, source_seq_len, d_model)
        x = self.pos_embedding(x)  # (batch_size, source_seq_len, d_model)

        for layer in self.layers:
            x = layer(x, mask)  # (batch_size, source_seq_len, d_model)

        x = self.layer_norm(x)  # (batch_size, source_seq_len, d_model)

        # x: (batch_size, source_seq_len, d_model)
        return x

class DecoderLayer(nn.Module):
    """ Decoder is made up of a self-attention layer, a encoder-decoder attention
        layer and a feed-forward layer.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

        self.attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.enc_attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.enc_attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.ff_layer = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.ff_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x, memory, src_mask, tgt_mask):
        """ x: (batch_size, target_seq_len, d_model)
            memory: (batch_size, source_seq_len, d_model)
            src_mask: (batch_size, 1, source_seq_len)
            tgt_mask: (batch_size, target_seq_len, target_seq_len)
        """
        # apply self-attention
        x1, _ = self.attn_layer(x, x, x, tgt_mask)  # (batch_size, target_seq_len, d_model)

        # apply residual connection followed by layer normalization
        x = self.attn_layer_norm(x + self.dropout(x1))  # (batch_size, target_seq_len, d_model)

        # apply encoder-decoder attention
        # memory is the output from encoder block (encoder states)
        x1, attn = self.enc_attn_layer(x, memory, memory, src_mask)

        # x1: (batch_size, target_seq_len, d_model)
        # attn: (batch_size, n_heads, target_seq_len, source_seq_len)

        # apply residual connection followed by layer normalization
        x = self.attn_layer_norm(x + self.dropout(x1))  # (batch_size, target_seq_len, d_model)

        # apply position-wise feed-forward
        x1 = self.ff_layer(x)  # (batch_size, target_seq_len, d_model)

        # apply residual connection followed by layer normalization
        x = self.ff_layer_norm(x + self.dropout(x1))  # (batch_size, target_seq_len, d_model)

        # x: (batch_size, target_seq_len, d_model)
        # attn: (batch_size, n_heads, target_seq_len, source_seq_len)
        return x, attn

class Decoder(nn.Module):
    """ Decoder block is a stack of N identical decoder layers.
    """

    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, pad_idx, dropout_rate=0.1, max_len=5000):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.pad_idx = pad_idx
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = PositionalEncoding(d_model, dropout_rate, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, x, memory, src_mask, tgt_mask):
        """ x: (batch_size, target_seq_len, d_model)
            memory: (batch_size, source_seq_len, d_model)
            src_mask: (batch_size, 1, source_seq_len)
            tgt_mask: (batch_size, target_seq_len, target_seq_len)
        """
        # apply positional encoding to token sequences
        x = self.tok_embedding(x)  # (batch_size, target_seq_len, d_model)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        for layer in self.layers:
            x, attn = layer(x, memory, src_mask, tgt_mask)  # (batch_size, target_seq_len, d_model)

        x = self.layer_norm(x)  # (batch_size, target_seq_len, d_model)

        # x: (batch_size, target_seq_len, d_model)
        # attn: (batch_size, n_heads, target_seq_len, source_seq_len)
        return x, attn

class Transformer(nn.Module):
    """ Transformer wrapper for encoder and decoder.
    """

    def __init__(self, encoder, decoder, generator, pad_idx):
        super(Transformer, self).__init__()
        self.pad_idx = pad_idx

        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator


    def get_pad_mask(self, x, pad_idx):
        """ x: (batch_size, seq_len)
        """
        x = (x != pad_idx).unsqueeze(-2)  # (batch_size, 1, seq_len)

        # x: (batch_size, 1, seq_len)
        return x


    def get_subsequent_mask(self, x):
        """ x: (batch_size, seq_len)
        """
        seq_len = x.size(1)
        subsequent_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1).astype(np.int8)  # (batch_size, seq_len, seq_len)
        subsequent_mask = (torch.from_numpy(subsequent_mask) == 0).to(x.device)  # (batch_size, seq_len, seq_len)


        # subsequent_mask: (batch_size, seq_len, seq_len)
        return subsequent_mask


    def forward(self, src, tgt):
        """ src: (batch_size, source_seq_len)
            tgt: (batch_size, target_seq_len)
        """
        # create masks for source and target
        src_mask = self.get_pad_mask(src, self.pad_idx)
        tgt_mask = self.get_pad_mask(tgt, self.pad_idx) & self.get_subsequent_mask(tgt)

        # src_mask: (batch_size, 1, seq_len)
        # tgt_mask: (batch_size, seq_len, seq_len)

        # encode the source sequence
        enc_output = self.encoder(src, src_mask)  # (batch_size, source_seq_len, d_model)

        # decode based on source sequence and target sequence generated so far
        dec_output, attn = self.decoder(tgt, enc_output, src_mask, tgt_mask)

        # dec_output: (batch_size, target_seq_len, d_model)
        # attn: (batch_size, n_heads, target_seq_len, source_seq_len)

        # apply linear projection to obtain the output distribution
        output = self.generator(dec_output)  # (batch_size, target_seq_len, vocab_size)

        # output: (batch_size, target_seq_len, vocab_size)
        # attn: (batch_size, n_heads, target_seq_len, source_seq_len)
        return output, attn

class Generator(nn.Module):
    """ Linear projection layer for generating output distribution.
    """

    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)


    def forward(self, x):
        """ x: (batch_size, target_seq_len, d_model)
        """
        # apply linear projection followed by softmax to obtain output distribution
        x = self.proj(x)  # (batch_size, target_seq_len, vocab_size)
        output = F.log_softmax(x, dim=-1)  # (batch_size, target_seq_len)

        # output: (batch_size, target_seq_len)
        return output

def model_summary(model):
    print(model)
    print(f'# of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    print(f'# of non-trainable params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}')

#------------------ Translate work

def summary(sentence, model, tokenizer, device, max_len=50):
    model.eval()
    
    # Tokenize the source sentence
    src_tensor = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)["input_ids"].to(device)
    
    # Initialize the target sequence with the Start-Of-Sequence (SOS) token
    tgt_indices = [tokenizer.cls_token_id]
    
    for i in range(max_len):
        # Convert current target list into a tensor: shape (1, current_seq_len)
        tgt_tensor = torch.tensor([tgt_indices]).to(device)
        
        with torch.no_grad():
            # Forward pass through your custom Transformer
            output, _ = model(src_tensor, tgt_tensor)
            
        # Get the prediction for the last token in the sequence
        # output shape: (1, current_seq_len, vocab_size)
        pred_token = output[:, -1, :].argmax(dim=-1).item()
        
        # Append to our target list
        tgt_indices.append(pred_token)
        
        # Stop if the model predicts the End-Of-Sequence (EOS) token
        if pred_token == tokenizer.sep_token_id:
            break
            
    # Decode the generated IDs back into a string
    # Drop the initial SOS token at the start, and skip_special_tokens removes the EOS
    summarized_text = tokenizer.decode(tgt_indices[1:], skip_special_tokens=True)
    
    return summarized_text

class NoamOptim(object):
    """ Optimizer wrapper for learning rate scheduling.
    """
    
    def __init__(self, optimizer, d_model, factor, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.factor = factor
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.n_steps += 1
        lr = self.get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.optimizer.step()

    def get_lr(self):
        return self.factor * (
            self.d_model ** (-0.5)
            * min(self.n_steps ** (-0.5), self.n_steps * self.n_warmup_steps ** (-1.5))
        )

def train_fn(model, iterator, optimizer, criterion, clip=1.0):
    model.train()
    total_loss = 0
    steps = 0

    tk0 = tqdm(iterator, total=len(iterator), position=0, leave=True)

    for idx, batch in enumerate(tk0):
        
        source = batch['input_ids'].to(device)
        target = batch['labels'].to(device)

        # forward pass
        optimizer.zero_grad()
        
        output, _ = model(source, target[:, :-1])

        # calculate the loss
        loss = criterion(
            output.view(-1, output.size(-1)), 
            target[:, 1:].contiguous().view(-1)
        )
        total_loss += loss.item()
        steps += 1

        output = output.argmax(dim=-1)

        # backward pass
        loss.backward()

        # clip gradients
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        # update model parameters
        optimizer.step()

        tk0.set_postfix(loss=total_loss/steps)

    tk0.close()
    perplexity = np.exp(total_loss / len(iterator))

    return output, perplexity

import evaluate
from tqdm.auto import tqdm
from bert_score import BERTScorer

# Load the modern BLEU metric
metric = evaluate.load("sacrebleu")

def eval_fn(model, iterator, criterion, tokenizer):
    model.eval()
    total_loss = 0.0
    steps = 0
    
    hypotheses = []
    references = []

    tk0 = tqdm(iterator, total=len(iterator), position=0, leave=True)

    with torch.no_grad():
        for idx, batch in enumerate(tk0):

            source = batch['input_ids'].to(device)
            target = batch['labels'].to(device)

            # Forward pass
            output, _ = model(source, target[:, :-1])

            # Calculate the loss
            loss = criterion(
                output.view(-1, output.size(-1)),
                target[:, 1:].contiguous().view(-1)
            )
            total_loss += loss.item()
            steps += 1

            # Get the predicted token IDs
            predictions = output.argmax(dim=-1)

            # Convert Token IDs back to text strings
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            
            # The target needs to be shifted by 1 to match the output predictions
            decoded_targets = tokenizer.batch_decode(target[:, 1:], skip_special_tokens=True)

            # Store texts for BLEU calculation
            hypotheses.extend(decoded_preds)
            
            # SacreBLEU expects references to be a list of lists (in case of multiple valid translations)
            references.extend([[t] for t in decoded_targets])

            tk0.set_postfix(loss=total_loss/steps)

    tk0.close()
    
    # Calculate Perplexity
    perplexity = np.exp(total_loss / len(iterator))
    
    # Calculate BLEU score using the evaluate library
    bleu_result = metric.compute(predictions=hypotheses, references=references)
    bleu4 = bleu_result["score"]

    scorer = BERTScorer(lang="en", rescale_with_baseline=True, device=device)
    P, R, F1 = scorer.score(hypotheses, references)   # lists of str, same length
    bertscore_f1 = F1.mean().item()

    print(f"Summary F1 Score (bert): = {bertscore_f1}")

    return hypotheses, perplexity, bleu4

#----------------------------   Training

# hyperparameters
INPUT_SIZE = len(tokenizer)
OUTPUT_SIZE = len(tokenizer)
HIDDEN_SIZE = 512
N_LAYERS = 6
N_HEADS = 8
FF_SIZE = 2048
DROPOUT_RATE = 0.1
N_EPOCHS = 10
CLIP = 1.0

encoder = Encoder(INPUT_SIZE, HIDDEN_SIZE, N_LAYERS, N_HEADS, FF_SIZE, PAD_IDX, DROPOUT_RATE)
decoder = Decoder(OUTPUT_SIZE, HIDDEN_SIZE, N_LAYERS, N_HEADS, FF_SIZE, PAD_IDX, DROPOUT_RATE)
generator = Generator(HIDDEN_SIZE, OUTPUT_SIZE)

model = Transformer(encoder, decoder, generator, PAD_IDX).to(device)
model_summary(model)

optimizer = NoamOptim(
    optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9),
    model.encoder.d_model, 2, 4000
)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

best_bleu4 = float('-inf')
es_patience = 100
patience = 0
model_path = 'model.pth'


for epoch in range(0, N_EPOCHS + 1):
    # one epoch training
    _, train_perplexity = train_fn(model, train_iterator, optimizer, criterion, CLIP)

    # one epoch validation
    _, valid_perplexity, valid_bleu4 = eval_fn(model, valid_iterator, criterion, tokenizer)

    print(f'Epoch: {epoch}, Train perplexity: {train_perplexity:.4f}, Valid perplexity: {valid_perplexity:.4f}, Valid BLEU4: {valid_bleu4:.4f}')

    random_idx = np.random.randint(0, len(train_dataset) - 1)
    
    # Extract the source and target token IDs
    sample_source_ids = train_dataset[random_idx]['input_ids']
    sample_target_ids = train_dataset[random_idx]['labels']
    
    # Decode them back into readable text (removing pad/sos/eos tokens)
    random_source_text = tokenizer.decode(sample_source_ids, skip_special_tokens=True)
    ground_truth_text = tokenizer.decode(sample_target_ids, skip_special_tokens=True)
    
    # Generate the model's translation
    model_translation = summary(random_source_text, model, tokenizer, device)

    print(f"\n--- Epoch {epoch} Random Translation Sample ---")
    print(f"Source:      {random_source_text}")
    print(f"Target:      {ground_truth_text}")
    print(f"Translation: {model_translation}")
    print("-" * 60 + "\n")
    
    # early stopping
    is_best = valid_bleu4 > best_bleu4
    if is_best:
        print(f'BLEU score improved ({best_bleu4:.4f} -> {valid_bleu4:.4f}). Saving Model!')
        best_bleu4 = valid_bleu4
        patience = 0
        torch.save(model.state_dict(), model_path)
    else:
        patience += 1
        print(f'Early stopping counter: {patience} out of {es_patience}')
        if patience == es_patience:
            print(f'Early stopping! Best BLEU4: {best_bleu4:.4f}')
            break
    print(f'{"_"*120}')

# evaluate the model on test data
model.load_state_dict(torch.load(model_path, map_location=device))

print('Evaluating the model on test data ...')
_, test_perplexity, test_bleu4 = eval_fn(model, test_iterator, criterion, tokenizer)
print(f'Test perplexity: {test_perplexity:.4f}, Test BlEU4: {test_bleu4:.4f}')

def translate_and_get_attention(sentence, model, tokenizer, device, max_len=50):
    model.eval()
    
    # Tokenize the source sentence
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    src_tensor = inputs["input_ids"].to(device)
    
    # Extract source tokens as strings for plotting
    src_tokens = tokenizer.convert_ids_to_tokens(src_tensor[0])
    
    # Initialize the target sequence with the SOS token (cls_token for BERT models)
    tgt_indices = [tokenizer.cls_token_id]
    
    attention_scores = None
    
    for i in range(max_len):
        tgt_tensor = torch.tensor([tgt_indices]).to(device)
        
        with torch.no_grad():
            # Get BOTH the output distribution and the attention matrix
            output, attn = model(src_tensor, tgt_tensor)
            
        # Get the predicted token
        pred_token = output[:, -1, :].argmax(dim=-1).item()
        tgt_indices.append(pred_token)
        
        # Save the attention scores from this step
        # attn shape: (batch_size, n_heads, current_tgt_len, src_len)
        attention_scores = attn
        
        # Stop if EOS token is generated
        if pred_token == tokenizer.sep_token_id:
            break
            
    # Decode the final text
    translated_text = tokenizer.decode(tgt_indices[1:], skip_special_tokens=True)
    
    # Get the target tokens as strings for plotting the Y-axis
    tgt_tokens = tokenizer.convert_ids_to_tokens(tgt_indices)
    
    # Clean up the attention tensor
    # Remove the batch dimension (since batch_size=1) and move to CPU for plotting
    # Final shape: (n_heads, target_seq_len, source_seq_len)
    attention_scores = attention_scores.squeeze(0).cpu()
    
    return translated_text, attention_scores, src_tokens, tgt_tokens

def plot_attention(sentence, model, tokenizer, device):
    # Get the translation and attention data using our previous function
    translation, attention, src_tokens, tgt_tokens = translate_and_get_attention(
        sentence, model, tokenizer, device
    )
    
    print(f"Source:      {sentence}")
    print(f"Translation: {translation}")
    
    # attention shape is (n_heads, target_len, source_len)
    n_heads = attention.shape[0]
    
    cols = 4 if n_heads >= 4 else n_heads
    rows = math.ceil(n_heads / cols)
    
    # Create the subplot figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    if n_heads > 1:
        axes = axes.flatten()
    else:
        axes = [axes] # Handle the edge case of a 1-head model
        
    for i in range(n_heads):
        ax = axes[i]
        attn_matrix = attention[i].numpy()
        
        # Create the heatmap on the specific subplot (ax)
        sns.heatmap(
            attn_matrix, 
            cmap="Blues", 
            cbar=False,
            xticklabels=src_tokens, 
            yticklabels=tgt_tokens,
            ax=ax
        )
        
        ax.set_title(f"Head {i + 1}")
        
        # Rotate X labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


        # Only show the Y-axis labels (target words) on the far-left plots
        if i % cols != 0:
            ax.set_yticks([])
            
        # Only show the X-axis labels (source words) on the bottom-row plots
        if i < (rows - 1) * cols:
            ax.set_xticks([])

    for i in range(n_heads, len(axes)):
        fig.delaxes(axes[i])
        
    plt.suptitle("Multi-Head Attention Alignment", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

example_idx = 1

source = train_df.iloc[example_idx]["source"]
target = train_df.iloc[example_idx]["target"]
print(f"Source: {source}")
print(f"Target: {target}")

plot_attention(source, model, tokenizer, device)