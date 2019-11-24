    from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score,accuracy_score,precision_score
import pickle
import numpy as np


clean_data = pd.read_csv("afterNormalize.csv")

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english') 
tfidf = tfidf_vectorizer.fit_transform(clean_data['tidy']) 

train_d = tfidf[:7000,:]
test_d = tfidf[7000:,:]

x_train,x_valid,y_train,y_valid = train_test_split(train_d, clean_data['score'][:7000],
                                                   random_state=42,test_size = 0.3)
lreg = LogisticRegression()
lreg.fit(x_train,y_train)
pickle.dump(lreg,open('tfIdfModel.sav','wb')) 

prediction = lreg.predict_proba(x_valid)
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0 
prediction_int = prediction_int.astype(np.int) 
print(prediction_int)
print("Accuracy= ", accuracy_score(y_valid,prediction_int))
print("Precision= ", precision_score(y_valid,prediction_int))
print("F1 score= ",f1_score(y_valid, prediction_int)) # calculating f1 score for the validation set


test_pred = lreg.predict_proba(test_d)
test_pred_int = test_pred[:,1] >= 0.3 
test_pred_int = test_pred_int.astype(np.int) 
clean_data['score'][7000:] = test_pred_int 
submission = clean_data[['id','score','review']] [7000:]
submission.to_csv('outputTFIDF_LR.csv', index=False) # writing data to a CSV file