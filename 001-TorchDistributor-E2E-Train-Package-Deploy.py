# Databricks notebook source
# MAGIC %md 
# MAGIC # 001-Train-Model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation

# COMMAND ----------

# MAGIC %pip install lightning==2.0.0

# COMMAND ----------

from pytorch_lightning import Trainer
from pyspark.sql.types import *
from pyspark.sql.functions import *
import mlflow
import mlflow.spark
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from mlflow.exceptions import RestException
## import the TorchDistributor library
from pyspark.ml.torch.distributor import TorchDistributor

# COMMAND ----------

class ReviewsDataModule(pl.LightningDataModule):
  def __init__(self, data_dir, batch_size=32):
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size

  # STEP 1: CLEAN AND PREP DATA FROM CSV
  def clean_text(self, text):
    import re
    # Remove special characters, numbers, and extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text

  def prep_raw_data_for_tokenization (self):
    import pandas as pd
    data_df = spark.read.csv(self.data_dir, header=True, inferSchema=True).where("Sentiment NOT LIKE '%the damage%'") \
                    .withColumn("SentimentScore", when(col("Sentiment") == "positive", 0).when(col("Sentiment") == "negative", 1).otherwise(2))
    prompts = [prompt[0] for prompt in data_df.select("Sentence").collect()]
    labels = [label[0] for label in data_df.select("Sentiment").collect()]
    jsons = ",".join([json.dumps({"full_text": prompt, "label":labels[i], "index":i}) for i, prompt in enumerate(prompts)])
    prepped_df = pd.read_json(jsons, lines=True)
    prepped_df["full_text"] = prepped_df["full_text"].apply(self.clean_text)
    return prepped_df

  # STEP 2: TOKENIZE AND BUILD VOCAB FROM CLEANED DATA
  def tokenize_and_build_vocab (self, prepped_df):
    from collections import Counter
    from itertools import chain
    from nltk.tokenize import word_tokenize
    nltk.download('punkt')
    tokenized_prompts = [word_tokenize(prompt) for prompt in prepped_df["full_text"]]
    word_counts = Counter(chain(*tokenized_prompts))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    vocab.update({word: i + 2 for i, (word, _) in enumerate(word_counts.most_common())})
    return tokenized_prompts, vocab

# COMMAND ----------

username = spark.sql("SELECT current_user()").collect()[0][0]
DB_NAME = "sentiment_analysis_demodb"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
sentiment_df = spark.read.csv(f"file:///Workspace/Repos/{username}/sentiment-analysis-demo-dbricks/data/data.csv", header=True, inferSchema=True).where("Sentiment NOT LIKE '%the damage%'").withColumn("SentimentScore", when(col("Sentiment") == "positive", 0).when(col("Sentiment") == "negative", 1).otherwise(2))
promptsDF, labelsDF = sentiment_df.select("Sentence"), sentiment_df.select("SentimentScore")

# COMMAND ----------

import nltk
from collections import Counter
from itertools import chain
nltk.download('punkt')

# tokenize the cleaned output
def tokenize_and_build_vocab (reviews_df):
  from nltk.tokenize import word_tokenize
  tokenized_reviews = [word_tokenize(review) for review in reviews_df["full_text"]]
  word_counts = Counter(chain(*tokenized_reviews))
  vocab = {'<PAD>': 0, '<UNK>': 1}
  vocab.update({word: i + 2 for i, (word, _) in enumerate(word_counts.most_common())})
  return tokenized_reviews, vocab

# COMMAND ----------

## Convert tokenized vocab into word indicies so that it can be used as input data for the PyTorch model
tokenized_reviews, vocab = tokenize_and_build_vocab(reviews_df)
encoded_reviews = [[vocab.get(token, vocab['<UNK>']) for token in review] for review in tokenized_reviews]

# COMMAND ----------

import numpy as np
# get the max_length that is appropriate for this dataset of movie reviews
# by getting the 95th percentile length of reviews in the dataset
# Assuming tokenized_reviews is a list of tokenized movie reviews
review_lengths = [len(review) for review in tokenized_reviews]
max_length = int(np.percentile(review_lengths, 95))
# Finally, convert the tokens into word indicies so that it  can be used as input data for the PyTorch model
import torch
from torch.nn.utils.rnn import pad_sequence

# Assuming 'encoded_reviews' is a list of lists containing the word indices for each movie review
# Convert each movie review (list of word indices) to a PyTorch tensor
encoded_reviews_tensors = [torch.tensor(review, dtype=torch.long) for review in encoded_reviews]

# Pad the sequences using pad_sequence
padded_reviews = pad_sequence(encoded_reviews_tensors, batch_first=True, padding_value=vocab['<PAD>'])

# If you want to truncate the sequences to a fixed length, you can do this manually
padded_reviews = padded_reviews[:, :max_length]

# COMMAND ----------

# split the datasets into train, test, validate datasets
X_train, X_test, y_train, y_test = train_test_split(padded_reviews, reviews_df['label'], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# COMMAND ----------

from torch.utils.data import DataLoader, Dataset
# Create train, validation, and test data
train_data = list(zip(X_train, y_train))
val_data = list(zip(X_val, y_val))
test_data = list(zip(X_test, y_test))

# Create DataLoader instances
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the Model and PyFunc Wrapper

# COMMAND ----------

import torch
import torch.nn as nn
import pytorch_lightning as pl

class SentimentModel(pl.LightningModule):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers, dropout=0.5):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient()
username = spark.sql("SELECT current_user()").collect()[0][0]
experiment_name = "sentiment_analysis_ex"
try:
  experiment_id = client.create_experiment(f"/Users/{username}/{experiment_name}/")
except Exception as e:
  print(e)
mlflow.set_experiment(f"/Users/{username}/{experiment_name}/")

# COMMAND ----------

conda_env = {
    "channels": ["defaults"],
    "dependencies": [
      "python=3.10.6",
      "pip",
      {"pip": [
          "mlflow<3,>=2.2",
          "accelerate==0.16.0",
          "attrs==21.4.0",
          "boto3==1.24.28",
          "cffi==1.15.1",
          "cloudpickle==2.0.0",
          "configparser==5.2.0",
          "defusedxml==0.7.1",
          "dill==0.3.4",
          "fsspec==2022.7.1",
          "googleapis-common-protos==1.56.4",
          "ipython==8.10.0",
          "lightning-utilities==0.8.0",
          "pandas==1.4.4",
          "pytorch-lightning==2.0.2",
          "rich==13.3.5",
          "scipy==1.9.1",
          "tensorflow==2.11.0",
          "torch==1.13.1",
          "torchmetrics==0.11.4",
          "torchvision==0.14.1",
          "transformers==4.26.1"
        ],
      },
    ],
    "name": "mlflow-env"
}

# COMMAND ----------

import mlflow.pyfunc


class SentimentModelWrapper(mlflow.pyfunc.PythonModel):
  def load_context (self, context):
      import json
      # read the artifact that has a key-value pair telling the DBFS location of the JSON file
      with open(context.artifacts["model-deps"], "r") as f:
        model_deps = json.load(f)
      # once you have the json file location and you have opened it, deserialize it into a python dictionary
      model_deps_json = json.loads(model_deps)
      model_path = model_deps_json['pytorch_model']
      self.model = mlflow.pytorch.load_model(model_path, map_location=torch.device('cpu')).eval()
      self.max_length = int(model_deps_json['max_length'])
      self.vocab = json.loads(model_deps_json["vocab"])

  def generate_tensor (self, new_review, vocab, max_length):
    import torch
    from nltk.tokenize import word_tokenize
    # Assuming 'new_review' is a string containing the new movie review text
    tokenized_review = word_tokenize(new_review)

    # Convert words to indices using the vocabulary
    indexed_review = [vocab.get(word, vocab['<UNK>']) for word in tokenized_review]

    # Pad or truncate the sequence to the same length as your training data (max_length)
    padded_review = indexed_review[:max_length] + [vocab['<PAD>']] * (max_length - len(indexed_review))

    # Convert the input to a PyTorch tensor and add a batch dimension
    input_tensor = torch.tensor(padded_review, dtype=torch.long).unsqueeze(0)
    return input_tensor

  def predict (self, context, model_input):
    import torch
    import json
    import torch.nn as nn
    with torch.no_grad():
      # get the prompt / value that is encapsulated in the pd.DataFrame 
      # for conversion to a Tensor. The value is the actual string prompt
      # that will be tokenized and converted into a tensor
      raw_value_for_tensor_conversion = model_input.iloc[0]["input"]
      # pre-process string to Tensor
      input_tensor = self.generate_tensor(raw_value_for_tensor_conversion, self.vocab, self.max_length)
      # pass tensor to model to get prediction probabilities
      prediction = self.model(input_tensor)
      probabilities = torch.softmax(prediction, dim=-1)
      # get the predicted class
      predicted_class = torch.argmax(probabilities).item()
      # convert predicated class to label
      sentiment_labels = {0: "positive", 1: "negative", 2:"neutral"}
      predicted_sentiment = sentiment_labels[predicted_class]
      # returned labeled prediction
      return json.dumps({"prediction": predicted_sentiment})



# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the Model

# COMMAND ----------

import random, os
import pickle
from mlflow.models.signature import infer_signature
exp_run_id = str(random.randint(10000,50000))

with mlflow.start_run(nested=True, run_name=f"sentiment-analysis-model-{exp_run_id}") as run:
  from pytorch_lightning import Trainer
  # Auto log all MLflow entities
  mlflow.pytorch.autolog()
  # log run id
  # Define the model's hyperparameters
  vocab_size = len(vocab)
  embed_dim = 100
  hidden_dim = 128
  output_dim = 3  # positive, neutral, negative classifications
  num_layers = 1
  # Create the model
  model = SentimentModel(vocab_size, embed_dim, hidden_dim, output_dim, num_layers)
  # Create the Trainer object
  trainer = Trainer(max_epochs=5, accelerator="gpu", devices=[0])
  # Train the model
  trainer.fit(model, train_loader, val_loader)
  # Evaluate your model
  trainer.test(model, test_loader)
  # add signature and input_example
  input_df = pd.DataFrame([reviews_df.iloc[5836]["full_text"]], columns=["input"])
  signature = infer_signature(input_df)
  mlflow.pytorch.log_model(trainer.model, artifact_path="pytorch-model", pickle_module=pickle, signature=signature, input_example=input_df)
  # log artifacts, objects, and model
  run_id = run.info.run_id
  pytorch_model_uri = f"runs:/{run_id}/pytorch-model"
  artifacts = json.dumps({
    "pytorch_model": pytorch_model_uri,
    "max_length": max_length,
    "vocab": json.dumps(vocab) 
  })
  # makes sure the path in dbfs exists
  if not os.path.exists(f"/dbfs/users/{username}/sentiment_model/"):
    os.makedirs(f"/dbfs/users/{username}/sentiment_model/")
  # serialize and write values to JSON file to be read in during context load
  # or during predict / inference in the pyfunc model class
  with open(f"/dbfs/users/{username}/sentiment_model/artifacts.json", "w") as f:
    json.dump(artifacts, f)

  mlflow.pyfunc.log_model(artifacts = {"model-deps": f"/dbfs/users/{username}/sentiment_model/artifacts.json"}, python_model=SentimentModelWrapper(), artifact_path = "pytorch-model-pyfunc", conda_env=conda_env, signature=signature, input_example=input_df)

  print(pytorch_model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Test the Trained Model

# COMMAND ----------

uri = f"runs:/{run_id}/pytorch-model-pyfunc"
import mlflow.pyfunc
# add signature and input example to log_model
## triggers the load_context()
loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri=uri)
## calls the predict() function in the pyfunc
pred = loaded_pyfunc_model.predict(input_df)

# COMMAND ----------

pred

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Trained Model with MLFlow Model Registry

# COMMAND ----------

model_name = 'sentiment_analysis_model'
model_details = mlflow.register_model(model_uri=uri, name=model_name)
