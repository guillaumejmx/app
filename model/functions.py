#from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, BertTokenizer, BertModel
from transformers import AlbertTokenizerFast, AlbertForSequenceClassification, AlbertModel
from transformers import pipeline
from scipy.special import softmax
from afinn import Afinn
af = Afinn()
import pandas as pd
import numpy as np
import torch
import re
import sentencepiece

save_directory = './model/classification/AlBERT'
tokenizer = AlbertTokenizerFast.from_pretrained(save_directory)
model = AlbertForSequenceClassification.from_pretrained(save_directory)

##################
# CLASSIFICATION #
##################
def predict(tweet):

  # Tokenize Tweet
  encoded_input = tokenizer.encode(tweet, truncation=True, padding=True,return_tensors="pt")

  # Predict Tweet Classes
  output = model(encoded_input)

  labels = ['hate', 'offensive', 'neither']
  scores = output[0][0].detach().numpy()
  scores = softmax(scores)

  ranking = np.argsort(scores)
  ranking = ranking[::-1]
  predictions = []

  # To send to the app
  for i in range(scores.shape[0]):
      l = labels[ranking[i]]
      s = scores[ranking[i]]

      if l == "hate":
        hate = round(scores[ranking[i]], 2)

      if l == "offensive":
        offensive = round(scores[ranking[i]], 2)

      if l == "neither":
          neither = round(scores[ranking[i]], 2)

  predictions = [hate, offensive, neither]
  return predictions

#############
# FILL MASK #
#############
def fillmask(tweet):

    tokenizer = AlbertTokenizerFast.from_pretrained(save_directory)
    model = AlbertModel.from_pretrained("albert-base-v2")

    # With Bert
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #model = BertModel.from_pretrained("bert-base-uncased")

    # fonction that find words in common in the tweet and in the hateful words database
    def words_in_string(word_list, a_string):
        return set(word_list).intersection(a_string.split())

    # import the hateful words database
    text_file = open("./model/fill-masking/hate_words.txt", "r")
    lines = text_file.readlines()
    lines = [item.replace("\n", "") for item in lines]

    # tweet input
    tweet = tweet + " !!!"
    tweet  = tweet
    tweet_splited = re.findall(r"[\w']+|[.,!?;]", tweet)
    tweet = " ".join(list(map(str,tweet_splited)))
    words = []

    # reset result
    result = []

    # apply the function
    for word in words_in_string(lines, tweet):
        words.append(word)

    if len(words)>0:
        if len(words) == 1:
            tweet = tweet.replace(word, "[MASK]")

            # train the model
            encoded_input = tokenizer(tweet, return_tensors='pt')

            output = model(**encoded_input)

            # replace hateful words
            unmasker = pipeline('fill-mask', model='bert-base-uncased')

            res = []
            for dict in unmasker(tweet):
              res.append(dict["token_str"])

            score = [af.score(word) for word in res]
            top_words = sorted(range(len(score)), key=lambda i: score[i], reverse = True)[:3]
            top_3 = [res[i] for i in top_words]

            result = [word, top_3]

        else:
            sentiment_scores = [af.score(word) for word in words]
            worst = words[sentiment_scores.index(min(sentiment_scores))]
            tweet = tweet.replace(words[sentiment_scores.index(min(sentiment_scores))],"[MASK]")

                # train the model
            encoded_input = tokenizer(tweet, return_tensors='pt')

            output = model(**encoded_input)

              # replace hateful words
            unmasker = pipeline('fill-mask', model='bert-base-uncased')

            res = []
            for dict in unmasker(tweet):
              res.append(dict["token_str"])

            score = [af.score(word) for word in res]
            top_words = sorted(range(len(score)), key=lambda i: score[i], reverse = True)[:3]
            top_3 = [res[i] for i in top_words]

            result = [worst, top_3]

    else:
            return ""



    return result
