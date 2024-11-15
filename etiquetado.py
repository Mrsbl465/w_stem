from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Etiquetado con RoBERTa
roberta_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
tweets_df["roberta_sentiment"] = tweets_df["clean_text"].apply(lambda x: roberta_analyzer(x)[0]['label'])

# Etiquetado con VADER
vader_analyzer = SentimentIntensityAnalyzer()
tweets_df["vader_sentiment"] = tweets_df["clean_text"].apply(lambda x: vader_analyzer.polarity_scores(x)['compound'])

tweets_df.to_csv("tweets_labeled.csv", index=False)
