import pandas as pd

data  = pd.read_csv('amazon_reviews_sample.csv')
#get counts of positive and negative reviews
count = data["score"].value_counts()
#data["score"].value_counts().plot.bar()
#convert pandas.series to dataframe
bar_count = count.to_frame()
x = bar_count["score"].plot.bar(color=['r','b'])

clean_data = data
clean_data["tidy"] = clean_data["review"].str.replace("[^a-zA-Z#]"," ")
#print(clean_data.head(20))
#clean_data.to_csv("afterRemovingPuncNumSpecialChar.csv")
clean_data["tidy"] = clean_data["tidy"].apply(lambda x:' '.join([w for w in x.split() if len(w)>3]))
clean_data.to_csv("afterRemoveThreeLetterWords.csv")

token_tweet = clean_data["tidy"].apply(lambda x: x.split())
token_tweet.head()

from nltk.stem.porter import *
stemmer = PorterStemmer()
token_tweet = token_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
token_tweet.head()

for i in range(len(token_tweet)):
    token_tweet[i] = ' '.join(token_t  weet[i])
clean_data["tidy"] = token_tweet
clean_data.to_csv("afterNormalize.csv")