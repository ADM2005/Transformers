import math
import os
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
spacy_english = spacy.load("en_core_web_sm") # Use "python -m spacy download en_core_web_sm" in terminal

### Dataset Used: IMDB Dataset of 50K Movie Reviews 

### Link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>": 2, "<UNK>": 3}

        self.freq_threshold = freq_threshold
    
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower()for tok in spacy_english.tokenizer(text)]
    
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class IMDB_Dataset(Dataset):
    def __init__(self, src, transform=None, freq_threshold=5, nrows=5000, skiprows=0):
        self.df = pd.read_csv(src, nrows=nrows, skiprows=skiprows)
        self.transform = transform

        self.reviews = self.df["review"]
        self.sentiments = self.df["sentiment"]
        
        # Initialize and Build Vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.reviews.tolist())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        review = self.reviews[index]
        sentiment = [1, 0] if self.sentiments[index] == "positive" else [0, 1]
        numericalized_review = [self.vocab.stoi["<SOS>"]]
        numericalized_review += self.vocab.numericalize(review)
        numericalized_review.append(self.vocab.stoi["<EOS>"])

        return torch.Tensor(numericalized_review).to(torch.int32),  sentiment
    
    def numericalize(self, text):
        numericalized = [self.vocab.stoi["<SOS>"]]
        numericalized += self.vocab.numericalize(text)
        numericalized.append(self.vocab.stoi["<EOS>"])
        return numericalized

class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    def __call__(self, batch):
        reviews = [item[0] for item in batch]
        sentiments = [item[1] for item in batch]
        reviews = pad_sequence(reviews, batch_first=False, padding_value=self.pad_idx)
        return reviews, torch.Tensor(sentiments)

def get_loader(
    root_folder,
    transform,
    batch_size=3,
    shuffle=True,
    pin_memory=True
):
    dataset = IMDB_Dataset(root_folder, transform=transform)
    
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=Collate(pad_idx=pad_idx)
    )

    return loader

filepath = ""
dataloader = get_loader(filepath, None)
class ReviewClassifier(nn.Module):
    def __init__(self, vocab_size, max_length, d_model, dim_feedforward, dropout=0.1, layers=6, n_heads=8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.emb_pos = nn.Embedding(max_length, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, layers)

        self.classifier = nn.Sequential(nn.Linear(d_model, 2))



    @staticmethod
    def positions(x):
        # Shape sentences * batch
        return torch.LongTensor([list(range(len(i))) for i in x]).to(device)

    def forward(self, x):
        word_emb = self.emb(x)
        pos_emb = self.emb_pos(self.positions(x))
        x = word_emb + pos_emb
        x = self.encoder(x)
        x = x.mean(dim=0)
        x = self.classifier(x)
        return x



epochs = 8
lr = 0.01
model = ReviewClassifier(len(dataloader.dataset.vocab.stoi), 50, 128, 2048).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

with torch.no_grad():
    print(F.softmax(model(torch.LongTensor([dataloader.dataset.numericalize("This movie is really bad")]).T.to(device))))

for epoch in range(epochs):
    epoch_loss = 0
    correct=0
    for idx, (reviews, labels) in enumerate(dataloader):
        reviews = reviews[:, :50].to(device)
        labels = torch.Tensor(labels).to(device)
        predictions = model(reviews)
        loss = criterion(predictions, labels)
        epoch_loss += float(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"{epoch+1}/{epochs}, loss={epoch_loss}")



print(F.softmax(model(torch.LongTensor([dataloader.dataset.numericalize("This movie is really bad")]).T.to(device))))
