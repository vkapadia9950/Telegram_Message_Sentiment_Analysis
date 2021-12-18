# -*- coding: utf-8 -*-
"""

# Telegram Message Sentiment Analysis (**Vishal Kapadia**)
"""

# Installing VaderSentiment
# !pip install vaderSentiment

# Installing Spacy_langdetect
# !pip install spacy_langdetect

# Installing Plotly
# !pip install plotly

# Installing tqdm for Progress Monitoring
# !pip install tqdm

# Importing necessary libraries
from tqdm import tqdm
for i in tqdm(range(1), desc="Importing Required Libraries"):
    import pandas as pd
    import numpy as np
    import re
    import json
    from collections import defaultdict
    import nltk
    from nltk.corpus import stopwords
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from nltk.stem.porter import *
    import warnings
    import csv
    import datetime
    from spacy_langdetect import LanguageDetector
    import spacy
    import plotly.io as pio
    import plotly.graph_objects as go
    warnings.filterwarnings("ignore")

# Reading the JSON File
for i in tqdm(range(1), desc="Reading Data from JSON File"):
    f = open('telegram.json')
    data = json.load(f)

# Extracting Telegram Messages in English Language
messages = []
date = []
lang = spacy.load('en')
lang.add_pipe(LanguageDetector(), name='language_detector', last=True)
for i in tqdm(range(len(data["messages"])),
              desc="Extracting Telegram Messages in English Language"):
    if isinstance(data["messages"][i]['text'], str):
        if lang(data["messages"][i]['text'])._.language['language'] == 'en':
            messages.append(data["messages"][i]['text'])
            date.append(data["messages"][i]['date'][0:10])

# Extracting Telegram Messages which contains "SHIB" and "DOGE"
messages_final = []
date_final = []

for i in tqdm(
        range(
            len(messages)),
        desc="Extracting Telegram Messages which contains SHIB and DOGE"):
    if re.search(
        "shib",
        messages[i],
        re.IGNORECASE) or re.search(
        "doge",
        messages[i],
            re.IGNORECASE):
        messages_final.append(messages[i])
        date_final.append(date[i])

# Creating DataFrame which contains telegram messages alongwith date when
# message was posted
telegram_json = {}
telegram_json['date'] = date_final
telegram_json['messages'] = messages_final
telegram_msg = pd.DataFrame(telegram_json)
telegram_msg.to_csv('telegram_msg.csv')

telegram_msg.to_csv('telegram_msg.csv')

# To Display All Text
pd.set_option('display.max_colwidth', None)

"""# **Messages Pre-Processing for Calculating Sentiment Score**

**Extracting Stopwords**
"""

# Downloading Stopwords
nltk.download('stopwords')

# Load English Stop Words
stopword = stopwords.words('english')
# print("Stopwords:",stopword)

"""**Messages Cleaning Function**"""

# Removing RT Word from Messages
telegram_msg['messages'] = telegram_msg['messages'].str.lstrip('RT')
# Removing selected punctuation marks from Messages
telegram_msg['messages'] = telegram_msg['messages'].str.replace(":", '')
telegram_msg['messages'] = telegram_msg['messages'].str.replace(";", '')
telegram_msg['messages'] = telegram_msg['messages'].str.replace(".", '')
telegram_msg['messages'] = telegram_msg['messages'].str.replace(",", '')
telegram_msg['messages'] = telegram_msg['messages'].str.replace("!", '')
telegram_msg['messages'] = telegram_msg['messages'].str.replace("&", '')
telegram_msg['messages'] = telegram_msg['messages'].str.replace("-", '')
telegram_msg['messages'] = telegram_msg['messages'].str.replace("_", '')
telegram_msg['messages'] = telegram_msg['messages'].str.replace("$", '')
telegram_msg['messages'] = telegram_msg['messages'].str.replace("/", '')
telegram_msg['messages'] = telegram_msg['messages'].str.replace("?", '')
telegram_msg['messages'] = telegram_msg['messages'].str.replace("''", '')
# Lowercase
telegram_msg['messages'] = telegram_msg['messages'].str.lower()

# Message Clean Function


def msg_clean(msg):
    # Remove URL
    msg = re.sub(r'https?://\S+|www\.\S+', " ", msg)

    # Remove Mentions
    msg = re.sub(r'@\w+', ' ', msg)

    # Remove Digits
    msg = re.sub(r'\d+', ' ', msg)

    # Remove HTML tags
    msg = re.sub('r<.*?>', ' ', msg)

    # Remove HTML tags
    msg = re.sub('r<.*?>', ' ', msg)

    # Remove Stop Words
    msg = msg.split()

    msg = " ".join([word for word in msg if word not in stopword])

    return msg


# Applying Message Clean Function
for i in tqdm(range(1), desc="Cleaning Telegram Messages for Tokenizing"):
    telegram_msg['Clean Messages'] = telegram_msg['messages'].astype(
        str).apply(lambda x: msg_clean(x))

# Tokenize Data
for i in tqdm(range(1), desc="Generating Tokens"):
    tokenize_msg = telegram_msg['Clean Messages'].apply(lambda x: x.split())
    # tokenize_msg.head()

"""**Tokenization**"""

# Tokenize the Messages
for i in tqdm(range(len(tokenize_msg)), desc="Tokenizing Messages"):
    tokenize_msg[i] = ' '.join(tokenize_msg[i])
telegram_msg['Clean Messages'] = tokenize_msg
# telegram_msg.head()

"""# Generating Sentiment Score of Telegram Messages

---


"""

# Calculate Sentiment Scores
for i in tqdm(range(1), desc="Calculating Sentiment Scores"):
    analyser = SentimentIntensityAnalyzer()

    scores = []
    for sentence in telegram_msg['Clean Messages']:
        score = analyser.polarity_scores(sentence)
        scores.append(score)

    scores = pd.DataFrame(scores)
    telegram_msg['Compound'] = scores['compound']
    telegram_msg['Negative'] = scores['neg']
    telegram_msg['Neutral'] = scores['neu']
    telegram_msg['Positive'] = scores['pos']

# telegram_msg.to_csv("final_sentiments.csv")

# List of Dates from May 1, 2021 to May 15, 2021
dates = telegram_msg['date'].unique().tolist()

# Calculating Number of Messages in a Day and Total Sentiments of Messages
# in that data
sum_sentiments = [0] * len(dates)
count_msg_freq = [0] * len(dates)
for i in tqdm(
        range(
            telegram_msg.shape[0]),
        desc="Counting Total Messages Per Day"):
    current_date = telegram_msg['date'][i]
    count_msg_freq[dates.index(current_date)] += 1
    sum_sentiments[dates.index(current_date)] += telegram_msg['Compound'][i]

# Calculating Average Sentiments Per Day
avg_sentiments = []
for i in tqdm(
        range(
            len(dates)),
        desc="Calculating Average Sentiments Per Day"):
    avg_sentiments.append(sum_sentiments[i] / count_msg_freq[i])

"""# Exploratory Data Analysis for Sentiments"""

# Setting up Plot Renderer to PNG
for i in tqdm(range(1), desc="Setting up Plot Renderer to PNG"):
    png_renderer = pio.renderers["png"]
    png_renderer.width = 5000
    png_renderer.height = 1000

# Plot of Number of Telegram Message Per Day
for i in tqdm(range(1), desc="Plotting Number of Telegram Message Per Day"):
    fig = go.Figure(
        data=[go.Bar(x=dates, y=count_msg_freq)],
        layout_title_text="Number of Telegram Message Per Day"
    )
    fig.show()

# Plot of Average Sentiments of Telegram Message Per Day
for i in tqdm(
        range(1),
        desc="Plotting Average Sentiments of Telegram Message Per Day"):
    fig = go.Figure(
        data=[go.Bar(x=dates, y=avg_sentiments)],
        layout_title_text="Average Sentiments of Telegram Message Per Day"
    )
    fig.show()
