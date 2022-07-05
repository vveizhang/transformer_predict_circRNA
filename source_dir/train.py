import math
import argparse
import collections
import copy
import os
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataloader
from sklearn import metrics
from torch.nn  import functional as F
import torch.optim as  optim
from tqdm import tqdm
import random
import ast
from sklearn import metrics
import logging
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import  f1_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
CUDA_LAUNCH_BLOCKING=1

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

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

def Kmers(sequence): 
    Kmers=[]   
    for seq in sequence:
        Kmers.append(build_kmers(seq,5))
    return Kmers

def get_vocab(vocab_dir):
    vocab_csv = pd.read_csv(vocab_dir)
    src_vocab = dict(zip(vocab_csv.kmer,vocab_csv.num))
    return src_vocab

class MyDataset(Dataset):
    def __init__(self, df, src_vocab):
        self.df = df
        self.src_vocab = src_vocab
        self.seqs = df.kmers
        self.label = df.label

    def __getitem__(self, idx):
        seqs = [src_vocab[seq] for seq in self.df.iloc[idx,2]]
#         for seq in self.df.iloc[idx,2]:
#             seqs.append(self.src_vocab.get(seq))
        seqs = torch.tensor(seqs, dtype=torch.int64)
        label = self.df.iloc[idx,1]
        return seqs, label

    def __len__(self):
        return len(self.seqs)

#len(src_vocab) 1366

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

def calculateMetrics(ypred,ytrue):
  acc  = accuracy_score(ytrue,ypred)
  f1  = f1_score(ytrue,ypred)
  f1_average  = f1_score(ytrue,ypred,average="macro")
  return " f1 score: "+str(round(f1,3))+" f1 average: "+str(round(f1_average,3))+" accuracy: "+str(round(acc,3))

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = myTransformer.to(device)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

def input_fn(input_data, content_type= 'application/json'):
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
    src_vocab = get_vocab('https://circ-rna.s3.us-east-2.amazonaws.com/data/vocab.csv')
    tokens=[src_vocab[kmer] for kmer in kmers]
        #data=torch.tensor(tokens, dtype=torch.float32)
    return torch.tensor(tokens, dtype=torch.float32).to(device)
# Perform prediction on the deserialized object, with the loaded model

def predict_fn(input_object, model):
    with torch.no_grad():       
        return model(input_object.unsqueeze(0).to(device))

#Serialize the prediction result into the desired response content type
def output_fn(prediction, accept="text/plain"):
    #logger.info('Serializing the generated output.')
    result = np.round(prediction.cpu().item())
#     if result == 1.0:
#         response = "Your inqury sequence IS circRNAs"
#     else:
#         response = "Your inqury sequence IS NOT circRNAs"    
    return str(result)


# save model
def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, default="")
    parser.add_argument("--test_data_dir", type=str, default="")
    #parser.add_argument("--output_folder", type=str, default="")
    #parser.add_argument("--model_name", type=str, default="")
    #parser.add_argument("--n_class", type=int, default=2)
    #parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--vocab_dir", type=str, default="")
    #parser.add_argument("--embed_dim", type=int, default=200)
    # parser.add_argument("--seq_length", type=int, default=396)
    # parser.add_argument("--embed_pos", type=int, default=20)
    #parser.add_argument("--dim_model", type=int, default=200)
    #parser.add_argument("--drop_out", type=float, default=0.1)
    #parser.add_argument("--num_head", type=int, default=8)
    #parser.add_argument("--num_encoder", type=int, default=6)
    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    args, _ = parser.parse_known_args()

    df_train = pd.read_csv(args.train_data_dir)
    df_train=df_train[['seqs','label']]
    df_test = pd.read_csv(args.test_data_dir)
    df_test=df_test[['seqs','label']]
    
#     with open('source_dir/vocab.json', 'r') as fp:
#         src_vocab = json.load(fp)
    src_vocab = get_vocab(args.vocab_dir)
    pad_seqs_train = pad_sequences(df_train.seqs)
    df_train['kmers'] = Kmers(pad_seqs_train)
    #src_vocab_train = Vocab(df_train['kmers'])

    pad_seqs_test = pad_sequences(df_test.seqs)
    df_test['kmers'] = Kmers(pad_seqs_test)
    #src_vocab_test = Vocab(df_test['kmers'])

    batch_size=args.batch_size

    train_data = MyDataset(df_train, src_vocab)
    test_data = MyDataset(df_test, src_vocab)
#dataAll = MyDataset(df.kmers,df.label)
# train_data = MyDataset(df1.kmers,df1.label,)
# train_itr = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# test_data = MyDataset(df2.kmers,df2.label)
# test_itr = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# train_data, test_data=torch.utils.data.random_split(dataAll,(30467,19000))
    train_itr= DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_itr= DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    model = myTransformer.to(device) 
    optimizer = optim.Adagrad(myTransformer.parameters(),lr = 0.001)

    for i in range(args.epochs):
        trainpreds = torch.tensor([])
        traintrues = torch.tensor([])
        for  batch in train_itr:
            X = batch[0].to(device)
            y = batch[1].to(torch.float).to(device)
            myTransformer.zero_grad()
            pred = myTransformer(X).squeeze()
            trainpreds = torch.cat((trainpreds,pred.cpu().detach()))
            traintrues = torch.cat((traintrues,y.cpu().detach()))
            err = F.binary_cross_entropy(pred,y)
            err.backward()
            optimizer.step()
        err = F.binary_cross_entropy(trainpreds,traintrues)
        print("train BCE loss: ",err.item(),calculateMetrics(torch.round(trainpreds).numpy(),traintrues.numpy()))

        valpreds = torch.tensor([])
        valtrues = torch.tensor([])
        for batch in test_itr:
            X = batch[0].to(device)
            y = batch[1].to(torch.float).to(device)
            valtrues = torch.cat((valtrues,y.cpu().detach()))
            pred = myTransformer(X).squeeze().cpu().detach()
    # print(valtrues.shape)
            valpreds = torch.cat((valpreds,pred))
        err = F.binary_cross_entropy(valpreds,valtrues)
        print("validation BCE loss: ",err.item(),calculateMetrics(torch.round(valpreds).numpy(),valtrues.numpy()))
        with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
            torch.save(model.state_dict(), f)
