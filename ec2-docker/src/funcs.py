import streamlit as st
import pandas as pd
import numpy as np
import json
from torch.nn  import functional as F
import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

# myTransformer = TextTransformer().to(device)
# model = myTransformer.to(device)

@st.cache
# def load_model(path:str):
#     model = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=5, max_seq_len=128, gpt_model_name="gpt2")
#     model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
#     model.eval()
#     return model

def load_model(path:str):
    model = TextTransformer()    
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# def load_tokenizer():
#     # load tokenizer
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     tokenizer.padding_side = "left"
#     tokenizer.pad_token = tokenizer.eos_token
#     return tokenizer

def build_kmers(sequence, ksize):
    kmers = []    
    n_kmers = len(sequence) - ksize + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)
    return kmers

def Kmers(sequence): 
    Kmers=[]   
    for seq in sequence:
        Kmers.append(build_kmers(seq,5))
    return Kmers

def get_vocab(vocab_dir):
    vocab_csv = pd.read_csv(vocab_dir)
    src_vocab = dict(zip(vocab_csv.kmer,vocab_csv.num))
    return src_vocab

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
    src_vocab = get_vocab('https://sagemaker-us-east-2-411668307327.s3.us-east-2.amazonaws.com/circRNA/vocab.csv')
    tokens=[src_vocab[kmer] for kmer in kmers]
    input_object = torch.tensor(tokens, dtype=torch.float32).to(device)
    return input_object

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

