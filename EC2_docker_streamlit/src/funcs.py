import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
import os
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_sequences(seqs,max_length=400,unk_index=64):
    pad_seqs=[]
    for seq in seqs:
        if len(str(seq))<max_length:
            pad_seqs.append(str(seq) + "0" * (max_length - len(str(seq))))
        if len(str(seq))>=max_length:
            pad_seqs.append(seq[0:max_length])
            #     mid_index=max_length//2
            #     pad_seqs.append((seq[:mid_index]+seq[(len(seq)-(max_length-mid_index)):],each[1]))
            
    return pad_seqs

def build_kmers(sequence, ksize):
    kmers = []    
    n_kmers = len(sequence) - ksize + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)
    return kmers

def get_vocab(vocab_dir="https://www.dropbox.com/s/7ubm7ad2if5a17y/vocab.csv?dl=1"):
    vocab_csv = pd.read_csv(vocab_dir)
    src_vocab = dict(zip(vocab_csv.kmer,vocab_csv.num))
    return src_vocab

class TextTransformer(nn.Module):
  def __init__(self):
    super(TextTransformer,self).__init__()
    self.wordEmbeddings = nn.Embedding(1366,396)
    self.positionEmbeddings = nn.Embedding(396,20)
    self.transformerLayer = nn.TransformerEncoderLayer(416,2) 
    self.linear1 = nn.Linear(416,  64)
    self.linear2 = nn.Linear(64,  1)
    self.linear3 = nn.Linear(396,  16)
    self.linear4 = nn.Linear(16,  1)
    
  def forward(self,x):
    positions = (torch.arange(0,396).reshape(1,396) + torch.zeros(x.shape[0],396)).to(device) 
    # broadcasting the tensor of positions 
    sentence = torch.cat((self.wordEmbeddings(x.long()),self.positionEmbeddings(positions.long())),axis=2)
    attended = self.transformerLayer(sentence)
    linear1 = F.relu(self.linear1(attended))
    linear2 = F.relu(self.linear2(linear1))
    linear2 = linear2.view(-1,396) # reshaping the layer as the transformer outputs a 2d tensor (or 3d considering the batch size)
    linear3 = F.relu(self.linear3(linear2))
    out = torch.sigmoid(self.linear4(linear3))
    return out
myTransformer = TextTransformer().to(device)
model = myTransformer.to(device)

@st.cache
def load_model(model_dir="https://www.dropbox.com/s/dazbgx8igqdgew5/model.pth?dl=1"):    
    model = myTransformer.to(device)
    with open(model_dir, "rb") as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    return model.to(device)

# def load_tokenizer():
#     # load tokenizer
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     tokenizer.padding_side = "left"
#     tokenizer.pad_token = tokenizer.eos_token
#     return tokenizer

# def read_input(tokenizer, input_text):
#     fixed_text = " ".join(input_text.lower().split())
#     model_input = tokenizer(fixed_text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
#     mask = model_input['attention_mask'].cpu()
#     input_id = model_input["input_ids"].squeeze(1).cpu()
#     return input_id, mask

def read_input(input_data, content_type= 'application/json'):
    #if content_type == 'text/plain':
    input = json.loads(input_data)
    seq = input["text"]
    #input = bytes(input)
    seq = seq.upper().replace("U","T")
    if len(seq) < 400:
        seq =  str(seq) + "0" * (400 - len(str(seq)))
    if len(seq) >= 400:
        seq =  seq[0:400] 
    kmers = build_kmers(seq,5)
    src_vocab = get_vocab("https://www.dropbox.com/s/7ubm7ad2if5a17y/vocab.csv?dl=1")
    tokens=[src_vocab[kmer] for kmer in kmers]
        #data=torch.tensor(tokens, dtype=torch.float32)
    return torch.tensor(tokens, dtype=torch.float32).to(device)

# def run_model(model, input_id, mask):
#     classes = ["business", "entertainment", "sport", "tech", "politics"]
#     output = model(input_id, mask)
#     prob = torch.nn.functional.softmax(output, dim=1)[0]
#     _, indices = torch.sort(output, descending=True)
#     return {classes[idx]: prob[idx].item() for idx in indices[0][:5]}

def run_model(input_object, model):
    with torch.no_grad():       
        prediction = model(input_object.unsqueeze(0).to(device))
    result = np.round(prediction.cpu().item())
    return str(result)

#Serialize the prediction result into the desired response content type
# def output_fn(prediction, accept="text/plain"):
#     #logger.info('Serializing the generated output.')
#     result = np.round(prediction.cpu().item())
# #     if result == 1.0:
# #         response = "Your inqury sequence IS circRNAs"
# #     else:
# #         response = "Your inqury sequence IS NOT circRNAs"    
#     return str(result)