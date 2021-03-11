# Hateful Speech Checker
![banner](./static/img/back.png)
---
Summary:
1. [_How an NLP assistant is helping to cope with hate speech on social networks_](#how-an-NLP-assistant-is-helping-to-cope-with-hate-speech-on-social-networks)
2. [App preview](#app-preview)
3. [Classifier accuracy](#classifier-accuracy)
4. [Technologies used](#technologies-used)
---
## _How an NLP assistant is helping to cope with hate speech on social networks_
This project has been run in the context of a final examination for the [__MSc Artificial Intelligence & Business Analytics__](https://www.linkedin.com/groups/12518036/) of __Toulouse Business School__.
The problematic given was: _"How limiting hateful speech on Twitter?"_.

We have thought that to limit effectively the diffusion of agressive contents on __Social Network Sites (SNS)__ the platform should present how the content is perceived by the system to the user, highlight the strongest word and suggest recommendations. Doing so, the user is fully aware of the consequence of his content and can choose to conform his speech or publish it as it is. In both case, it produces insightful behavioural information on the SNS's user to classify them.

## App preview
Consider the model to be integrated into the Tweeter interface. When the user is writing his content our model is classifying the scores and identifying the strongest word in order to offer suggestion to replace it.

![app_preview](./static/img/preview.png)

The user will replace the first word __bitch__ by __guy__ and then the __hateful__ score will decrease to be more __agressive__. The model identifies a new word __fuck__. We can iterate as long as the model detects strong content.

## Classifier accuracy
One of our challenge was being able __to differentiate neutral, offensive and hateful language__. To do so, we have used a __transformers__ that we have trained with 2 datasets: _Davidson et al., 2017 and Fonta et al., 2018_.

We have trained HuggingFace's __DistilBERT__ (Sanh et al., 2019), Facebook's __RoBERTa__ (Liu et al., 2019), and Google's __AlBERT__ (Lan et al., 2019) which results are summarized below:
Transformer | DistilBERT | RoBERTa | AlBERT
--- | --- | --- | ---
Nbr of parameters | 67M | 125M | 12M
Hate F1-score | 0.86 | __0.87__ | 0.84
Offensive F1-score | 0.87 | 0.86 | 0.85
Neither F1-score | 0.90 | __0.91__ | 0.89
Accuracy | __0.88__ | __0.88__ | 0.86

Despite a lower accuracy we choose to deploy ALBERT in our app since it is the lightest model.

## Technologies used
Our app has involved a deep understanding of several NLP algorithms as our functions perform __sentiment analysis__, __classification__ and __fill mask__ tasks.
The languages and framework used are:
* Python 3.8.8
* Flask 1.1.2
* HTML / CSS
* JS
* PyTorch 1.7.1
