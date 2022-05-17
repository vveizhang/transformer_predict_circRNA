import argparse
import logging
import json
import os
import sys
import boto3
import torch
import torch.nn as nn
from torch.nn  import functional as F
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def get_vocab(vocab_dir):
    vocab_csv = pd.read_csv(vocab_dir)
    src_vocab = dict(zip(vocab_csv.kmer,vocab_csv.num))
    return src_vocab

def build_kmers(sequence, ksize):
    kmers = []    
    n_kmers = len(sequence) - ksize + 1
    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)
    return kmers

class TextTransformer(nn.Module):
  def __init__(self):
    super(TextTransformer,self).__init__()
    self.wordEmbeddings = nn.Embedding(1366,396)
    self.positionEmbeddings = nn.Embedding(396,20)
    self.transformerLayer = nn.TransformerEncoderLayer(416,8) 
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

myTransformer = TextTransformer()
myTransformer.to(device)


#Loads the model parameters from a model.pth file in the SageMaker model directory model_dir.
def model_fn(model_dir):
    #logger.info('Loading the model.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = myTransformer.to(device)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))
    # model.to(device).eval()
    #logger.info('Done loading model')
    return model.to(device)

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, request_content_type='text/csv'):
    # Deserializing input data
    #logger.info('Deserializing the input data.')
    assert request_content_type == 'application/json'
        # input_data = json.loads(request_body)
        # text = input_data["text"]
        #logger.info(f'Input text: {text}')
    input = json.loads(request_body)["inputs"]
    seq = input.upper()
    if len(seq) < 400:
        seq =  str(seq) + "0" * (400 - len(str(seq)))
    if len(seq) >= 400:
        seq =  seq[0:400] 
    kmers = build_kmers(seq,5)
    src_vocab = get_vocab('https://sagemaker-us-east-2-411668307327.s3.us-east-2.amazonaws.com/circRNA/vocab.csv')
    tokens=[src_vocab[kmer] for kmer in kmers]
    data=torch.tensor(tokens, dtype=torch.int64)
    return data

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    #logger.info('Generating prediction based on input parameters.')
#     classes = [0,1]
#     tok = input_data.to(device)
    # model = myTransformer.to(device)
    # model.eval()
    #output = model(tok )
    # prob = torch.nn.functional.softmax(output, dim=1)[0]
    # _, indices = torch.sort(output, descending=True)
    with torch.no_grad():
        prediction = model(input_object)
        return prediction

def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)


# Serialize the prediction result into the desired response content type
# def output_fn(prediction, response_content_type='application/json'):
#     logger.info('Serializing the generated output.')
#     result = prediction
#     if response_content_type == 'application/json':
#         response_body_str = json.dumps(result)
#         return response_body_str
#     raise Exception(f'Requested unsupported ContentType in Accept:{response_content_type}')
