import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
from tkinter import *
import GUI


def Train_ALL():
    #dataset=read_dataset('dataset/a1_RestaurantReviews_HistoricDump.tsv')
    dataset = read_dataset('dataset/googleplaystore_user_reviews.csv')
    clean_dataset=data_set_cleaning(dataset)
    y_test,y_pred=train(dataset,clean_dataset)
    #show model accuracy
    test_model(y_test,y_pred)

def read_dataset(d_name):
    dataset = pd.read_csv(d_name)
    dataset=dataset.dropna()
    return dataset


def data_set_cleaning(dataset):
    #nltk.download('stopwords')
    ps = PorterStemmer()

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    res = []

    for word in dataset["Translated_Review"] :
        review = re.sub('[^a-zA-Z]', ' ', word)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        res.append(review)
    return res


def train(dataset,clean_dataset):
    #print(clean_dataset)
    cv = CountVectorizer(max_features=250)
    X = cv.fit_transform(clean_dataset).toarray()
    Y = dataset["Sentiment"]
    print(dataset.shape)

    #save data BOW
    bow_path = 'ModelAndData/c1_BoW_Sentiment_Model.pkl'
    pickle.dump(cv, open(bow_path, "wb"))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    joblib.dump(classifier, 'ModelAndData/c2_Classifier_Sentiment_Model')

    #pridict custom sentence

    #########################
    # pridict
    y_pred = classifier.predict(X_test)
    return y_test,y_pred

def test_model(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))

def test_custom_data():
    clean_dataset=[]
    test = GUI.sentence_text.get("1.0", 'end-1c')
    clean_dataset.append(test)
    cv=pickle.load(open("ModelAndData/c1_BoW_Sentiment_Model.pkl","rb"))
    sentence_test = cv.transform(clean_dataset).toarray()
    #print(sentence_test.shape)
    classifier = joblib.load('ModelAndData/c2_Classifier_Sentiment_Model')
    y_pred = classifier.predict(sentence_test)
    GUI.result.config(text=y_pred)
    #dataset['predicted_label'] = y_pred.tolist()
    #print(dataset.head())
    #dataset.to_csv("saved_result/c3_Predicted_Sentiments_Fresh_Dump.csv", sep='\t',
    #               encoding='UTF-8', index=False)



