<p align="center">
<br>
</p>

# Train and deploy Transformer model using PyTorch on Amazon SageMaker to classify non-coding RNA

## Table of Contents

- [Train and deploy Transformer model using PyTorch on Amazon SageMaker to classify non-coding RNA](#train-and-deploy-transformer-model-using-pytorch-on-amazon-sagemaker-to-classify-non-coding-rna)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
    - [1.1 Transformers](#11-transformers)
    - [1.2 Amazon SageMaker](#12-amazon-sagemaker)
  - [1.3 circRNAs](#13-circrnas)
  - [2. Dataset](#2-dataset)
  - [3. Demo](#3-demo)
  - [4. Training and deployment of GPT-2 on SageMaker](#4-training-and-deployment-of-gpt-2-on-sagemaker)
    - [4.1. Create an Amazon SageMaker notebook instance](#41-create-an-amazon-sagemaker-notebook-instance)
    - [4.2. Training and deployment](#42-training-and-deployment)
    - [4.3. The code](#43-the-code)
    - [4.4. Use AWS lambda to invoke the SageMaker endpoint](#44-use-aws-lambda-to-invoke-the-sagemaker-endpoint)
  - [5. Use Postman to test the endpoint](#5-use-postman-to-test-the-endpoint)
  - [6. Containerizing and Deployment using Amazon EC2 and Docker](#6-containerizing-and-deployment-using-amazon-ec2-and-docker)
    - [6.1. Create an Amazon EC2 instance](#61-create-an-amazon-ec2-instance)
    - [6.2. Running Docker container in cloud](#62-running-docker-container-in-cloud)
  - [7. Summary](#7-summary)
  - [8. References](#8-references)
  - [Contact](#contact)

Text classification is a very common task in NLP. It can be used in many applications from spam filtering, sentiment analysis to customer support automation and news categorization. Using Deep Learning language models for large-scale text classification tasks has become quite popular in the industry recently, especially so with the emergence of [Transformers](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) in recent years. Because the size of these Transformer models are often too large to train on local machines, cloud computing platforms (e.g. [GCP](https://cloud.google.com/), [AWS](https://aws.amazon.com/), [Azure](https://azure.microsoft.com/), [Alibabacloud](https://us.alibabacloud.com/)) are commonly used. Therefore in this blog, I want to demonstrate how to train and deploy a fine-tuned GPT-2 model for text classification tasks using Amazon SageMaker.

## 1. Introduction

### 1.1 Transformers

[Transformers](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) are the building block of the current state-of-the-art NLP architecture. It is impossible to explain how transformers work in one paragraph here, but to sum it up, transformers uses a "self-attention" mechanism that computes a representation of a sequence by "learning" the relationship between words at different positions in a sentence. A typical transformers design contains two parts, **encoder** and **decoders**, both working as vectorized representation of word relationships.

<p align="center">
<img src="https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1-727x1024.png">
<br>
<em>The Encoder-Decoder Structure of the Transformer Architecture
Taken from</a> on <a href="https://arxiv.org/abs/1706.03762">"Attention Is All You Need"</a></em></p>

The Transformer architecture follows an encoder-decoder structure, but does not rely on recurrence and convolutions in order to generate an output. In a nutshell, the task of the encoder, on the left half of the Transformer architecture, is to map an input sequence to a sequence of continuous representations, which is then fed into a decoder. 

The decoder, on the right half of the architecture, receives the output of the encoder together with the decoder output at the previous time step, to generate an output sequence.

### 1.2 Amazon SageMaker

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) is a great tool to train and deploy deep learning models on cloud instances with a fully-managed infrastructure provided by AWS. Within minutes, you can build, train and deploy a model in a Jupyter Notebook and don't have to worry about environment setup, because it comes with many pre-built Conda environments and Docker containers. It's a huge life-saver for data scientists like me.

## 1.3 circRNAs

Circular RNAs are diverse RNA species that are found in all live forms from archaea to humans. Although, circRNAs have been discovered over 20 years ago they were initially dismissed for having low abundance or resulting from splicing errors. However, recent advancements in high throughput sequencing revealed the presence of circRNAs in mammalian cells, across various cell lines and many transcripts are abundant and stable. CircRNAs are generated through a mechanism known as back-splicing “tail” to “head” whereby an exon at the 3′end of a gene is back-spliced to an exon at the 5′end of the gene resulting in a circular RNA form. CircRNAs are dispersed throughout the genome. They can arise mainly from exons but circRNAs deriving from inter- or intragenic, and intronic regions as well as antisense sequences have been reported.

<p align="center">
<img src="https://www.frontiersin.org/files/Articles/445805/fphar-10-00428-HTML/image_m/fphar-10-00428-g001.jpg">
<br>
<em>Schematic representation of circRNAs generation and function. from https://doi.org/10.3389/fphar.2019.00428</em></p>

## 2. Dataset

The dataset we are going to use in this project is from [here](https://github.com/liuyunho/Circ-ATTEN-MIL).

CircRNAs sequences were extracted from the circRNADb database and other lncRNAs sequences were extracted from the GENCODE database (lincRNA, antisense, processed transcript, sense intronic, and sense overlapping), respectively. After removing sequences shorter than 200 nucleotides, got 31,939 circRNAs and 19,722 other lncRNAs. 


## 3. Demo

I built an [Online circRNA Classifier](http://3.142.92.255:8501/) using [Streamlit](https://streamlit.io/) running the trained model. You can input or paste any noncoding RNA sequence, the online app will predict if the noncoding RNA is a circRNA or not.

<p align="center">
<img src="/imgs/Streamlit.png">
<br>
<em>Image by Author</em></p>

## 4. Training and deployment of GPT-2 on SageMaker

### 4.1. Create an Amazon SageMaker notebook instance

Follow this [hands-on tutorial](https://aws.amazon.com/getting-started/hands-on/build-train-deploy-machine-learning-model-sagemaker/) from AWS to create an Amazon SageMaker notebook instance. Use "*transformer*" as the **instance name**, and "*ml.t3.medium*" as the **instance type**.

<p align="center">
<img src="/imgs/Notebook_instance.png">
<br>
<em>Image by Author</em></p>

### 4.2. Training and deployment

Run [this notebook](https://github.com/vveizhang/transformer_predict_circRNA/blob/main/sagemaker/AWS_circRNA_Transformer.ipynb) on SageMaker to train and deploy the transformer model. Read through it to get more details on the implementation.

### 4.3. The code

Since we are building and training a PyTorch model in this project, it is recommended by [**SageMaker Python SDK**](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#train-a-model-with-pytorch) to prepare a separate `train.py` script to construct and store model functions used by SageMaker. Since all the pretrained model are not suitable for DNA/RNA sequence analysis, we will write our own model.
```python
def pad_sequences(seqs,max_length=400,unk_index=64):
    pad_seqs=[]
    for seq in seqs:
        if len(str(seq))<max_length:
            pad_seqs.append(str(seq) + "0" * (max_length - len(str(seq))))
        if len(str(seq))>=max_length:
            pad_seqs.append(seq[0:max_length])           
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
```
The `pad_sequences` function will pad all the input sequence into the same length (max_length), the `build_kmers` and `Kmers` functions will be used to build the vocabulary. A DNA sequence can be viewed as a collection of k-mers by breaking the sequence into nucleotide substrings of length k, as illustrated in the Figure.

The `TextTransformer` class in *train.py* is responsible for building a classifier from the scratch. Instead of a positional encoding, I did a positional embedding here. So this model has two embedding steps: word embeddings and position embeddings.

```python
class TextTransformer(nn.Module):
  def __init__(self):
    super(TextTransformer,self).__init__()
    self.wordEmbeddings = nn.Embedding(vocab_size,seq_len)
    self.positionEmbeddings = nn.Embedding(seq_len,posEmbSize)
    self.transformerLayer = nn.TransformerEncoderLayer(seq_len+posEmbSize,2) 
    self.linear1 = nn.Linear(seq_len+posEmbSize,  64)
    self.linear2 = nn.Linear(64,  1)
    self.linear3 = nn.Linear(seq_len,  16)
    self.linear4 = nn.Linear(16,  1)
    
  def forward(self,x):
    positions = (torch.arange(0,seq_len).reshape(1,seq_len) + torch.zeros(x.shape[0],seq_len)).to(device) 
    # broadcasting the tensor of positions 
    sentence = torch.cat((self.wordEmbeddings(x.long()),self.positionEmbeddings(positions.long())),axis=2)
    attended = self.transformerLayer(sentence)
    linear1 = F.relu(self.linear1(attended))
    linear2 = F.relu(self.linear2(linear1))
    linear2 = linear2.view(-1,seq_len) # reshaping the layer as the transformer outputs a 2d tensor (or 3d considering the batch size)
    linear3 = F.relu(self.linear3(linear2))
    out = torch.sigmoid(self.linear4(linear3))
    return out
```

The `model_fn`,`input_fn`,`predict_fn`,`output_fn`,`save_model` functions in *train.py* will be responsible for the communications to the AWS sagemaker APIs to load the model weights/parameters, encode input, make predictions, output the predict results and save model weights/parameters.

```python
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = myTransformer.to(device)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

def input_fn(input_data, content_type= 'application/json'):
    input = json.loads(input_data)
    seq = input["text"]
    seq = seq.upper().replace("U","T")
    if len(seq) < 400:
        seq =  str(seq) + "0" * (400 - len(str(seq)))
    if len(seq) >= 400:
        seq =  seq[0:400] 
    kmers = build_kmers(seq,5)
    src_vocab = get_vocab('https://sagemaker-us-east-2-411668307327.s3.us-east-2.amazonaws.com/circRNA/vocab.csv')
    tokens=[src_vocab[kmer] for kmer in kmers]
    return torch.tensor(tokens, dtype=torch.float32).to(device)

def predict_fn(input_object, model):
    with torch.no_grad():       
        return model(input_object.unsqueeze(0).to(device))

def output_fn(prediction, accept="text/plain"):
    result = np.round(prediction.cpu().item())   
    return str(result)

# save model
def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
```

When the traning is finished. You will see the following:

<p align="center">
<img src="/imgs/sagemaker_endpoints.png">
<br>
<em>Image by Author</em></p>

Then deploy the trained model into the inference endpoint.

### 4.4. Use AWS lambda to invoke the SageMaker endpoint

Create an AWS lambda function `predict_circRNA_transformer` to invoke the inference endpoint. Below is the code for the lambda handler:
```python
import os
import io
import boto3
import json
import csv

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Body=json.dumps(event))
    result=response['Body'].read().decode()
    print(result)
    predicted_label = 'circRNA' if result == 1 else 'lincRNA'    
    return predicted_label
```

Use AWS REST API `predict_circRNA` to create an application program interface (API) that uses HTTP requests to access and use data. 

<p align="center">
<img src="/imgs/API_gateway_lambda.png">
<br>
<em>Image by Author</em></p>

## 5. Use Postman to test the endpoint
Now i have everything on AWS, next is to test the HTTP requests to do the prediction. Usually we use Postman to do the test.
<p align="center">
<img src="/imgs/postMan.png">
<br>
<em>Image by Author</em></p>

## 6. Containerizing and Deployment using Amazon EC2 and Docker

Docker is an open platform for developing, shipping, and running applications. Docker enables you to separate your applications from your infrastructure so you can deliver software quickly. With Docker, you can manage your infrastructure in the same ways you manage your applications. By taking advantage of Docker’s methodologies for shipping, testing, and deploying code quickly, you can significantly reduce the delay between writing code and running it in production. Here I also deployed the trained transformer model using Docker on Amazon EC2 instance.

### 6.1. Create an Amazon EC2 instance

Follow [this tutorial](https://docs.aws.amazon.com/efs/latest/ug/gs-step-one-create-ec2-resources.html) from AWS to create and launch an Amazon EC2 instance. A few customized settings for this project:

- In **Step 1: Choose an Amazon Machine Image (AMI)**, choose the **Deep Learning AMI (Ubuntu) AMI**. Using this image does introduce a bit of extra overhead, however, it guarantees us that git and Docker will be pre-installed which saves a lot of trouble.
- In **Step 2: Choose an Instance Type**, choose **t2.medium** to ensure we have enough space to build and run our Docker image.
- In **Step 6: Configure Security Group**, choose **Add Rule** and create a custom tcp rule for port **8501** to make our streamlit app publicly available.
- After clicking **Launch**, choose **Create a new key pair**, input "**ec2-transformer**", and click "**Download Key Pair**" to save `ec2-transformer.pem` key pair locally.

### 6.2. Running Docker container in cloud

After launching the EC2 instance, use SSH to connect to the instance:

```bash
ssh -i ec2-gpt2-streamlit-app.pem ec2-user@your-instance-DNS-address.us-east-1.compute.amazonaws.com
```

Then, copy my code into the cloud using `git`:

```bash
git clone https://github.com/vveizhang/transformer_predict_circRNA.git
```

Afterwards, go into the `ec2-docker` folder to build and run the image:

```bash
cd ec2-docker/
docker image build -t streamlit:circRNA-transformer .
```
The Dockerfile is:
```bash
# base image
FROM python:3.7.4-slim-stretch

# exposing default port for streamlit
EXPOSE 8501

# making directory of app
WORKDIR /streamlit:circRNA-transformer

# copy over requirements
COPY requirements.txt ./requirements.txt

# install pip then packages
RUN pip3 install -r requirements.txt

# copying all files over
COPY . .

# download model file
RUN apt-get update
RUN apt-get  -qq -y install wget
RUN wget -O ./model/transformer-model.pth "https://www.dropbox.com/s/dazbgx8igqdgew5/model.pth?dl=1"

# cmd to launch app when container is run
CMD streamlit run ./src/app.py

# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'
```
Then run the container from the build image:

```bash
docker container run -p 8501:8501 -d streamlit:circRNA-transformer
```

Now, you can access the Streamlit app at `[<EC2 public IP address>](http://3.142.92.255:8501/)`(EC2 public IP address can be found under "IPv4 Public IP" in the AWS console)!

## 7. Summary

All source code can be found in this Github Repo: [https://github.com/vveizhang/transformer_predict_circRNA](https://github.com/vveizhang/transformer_predict_circRNA)

The structure of this Github Repo shows here

```markdown
├── Readme.md                            # main notebook
├── sagemaker                            # AWS Sagemaker folder
│   ├── source_dir
│   │    ├── requirements.txt            # libraries needed
│   │    ├── lambda_handler.py           # AWS lambda function
│   │    └── train.py                    # PyTorch training/deployment script
│   └── AWS_circRNA_Transformer.ipynb    # AWS Sagemaker notebook
|  
└── ec2-docker                           # Streamlit app folder
    ├── Dockerfile                       # Dockerfile for the app (container)
    ├── requirements.txt                 # libraries used by app.py
    ├── model
    |   └── download_model.sh            # bash code to download the trained model weights
    └── src 
        ├── func.py                      # utility functions used by app.py    
        └── app.py                       # main code for the Streamlit app
```



## 8. References

- **Transformers**: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- **k-mers**: [Estimating the total genome length of a metagenomic sample using k-mers](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-5467-x)

- **Introduction of Circular RNAs**: [Biogenesis and Function of Circular RNAs in Health and in Disease](https://www.frontiersin.org/articles/10.3389/fphar.2019.00428/full)

- **Train and deploy models on AWS SageMaker**: [https://medium.com/@thom.e.lane/streamlit-on-aws-a-fully-featured-solution-for-streamlit-deployments-ba32a81c7460](https://medium.com/@thom.e.lane/streamlit-on-aws-a-fully-featured-solution-for-streamlit-deployments-ba32a81c7460)
- **Deploy Streamlit app on AWS EC2**: [https://medium.com/usf-msds/deploying-web-app-with-streamlit-docker-and-aws-72b0d4dbcf77](https://medium.com/usf-msds/deploying-web-app-with-streamlit-docker-and-aws-72b0d4dbcf77)

## Contact

- **Author**: Wei Zhang
- **Email**: [zwmc@hotmail.com](zwmc@hotmail.com)
- **Github**: [https://github.com/vveizhang](https://github.com/vveizhang)
- **Linkedin**: [https://www.linkedin.com/in/wei-z-76253523/](https://www.linkedin.com/in/wei-z-76253523/)
