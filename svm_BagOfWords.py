import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score,accuracy_score,precision_score
import pickle

clean_data = pd.read_csv("afterNormalize.csv")

vect = CountVectorizer(max_df=0.90,min_df=2,max_features=1000,stop_words='english')
fit_vect = vect.fit_transform(clean_data['tidy'])

train_d = fit_vect[:7000,:]
test_d = fit_vect[7000:,:]

x_train,x_valid,y_train,y_valid = train_test_split(train_d, clean_data['score'][:7000],
                                                   random_state=42,test_size = 0.3)

svc = svm.SVC(kernel='linear', C=1, probability=True).fit(x_train, y_train) 
pickle.dump(svc,open("svm.sav","wb"))

prediction = svc.predict_proba(x_valid) 
prediction_int = prediction[:,1] >= 0.3 
prediction_int = prediction_int.astype(np.int) 

print("Accuracy= ", accuracy_score(y_valid,prediction_int))
print("Precision= ", precision_score(y_valid,prediction_int))
print("F1 score= ",f1_score(y_valid, prediction_int))

test_pred = svc.predict_proba(test_d) 
test_pred_int = test_pred[:,1] >= 0.3 
test_pred_int = test_pred_int.astype(np.int) 
x = clean_data["score"][7000:]
clean_data["score"][7000:] = test_pred_int 
submission = clean_data[['id','score','review']][7000:] 
submission.to_csv('output_SVM.csv', index=False)

print("Accuracy= ", accuracy_score(x,test_pred_int))
print("Precision= ", precision_score(x,test_pred_int))
print("F1 score= ",f1_score(x,test_pred_int))
