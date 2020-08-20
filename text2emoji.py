import keras
from nltk import word_tokenize
import numpy as np
import pickle


def load_model():
    model =  keras.models.load_model("model.h5")
    pickle_in = open("word2index.pickle","rb")
    word2idx = pickle.load(pickle_in)
    return model , word2idx

def textpreprocessing(text,word2idx):
    text = word_tokenize(text.lower())
    X = np.zeros((128 , ))
    for i , word in enumerate(text[:128]):
        X[i] =  word2idx.get(word , 0)
    
    return X

def predict_emoji(X , model):
    res = np.argmax(model.predict_classes(X.reshape(1,128)))
    return res
    
def get_html_emoji(result):
    #return html code of emoji
    emoji2code = {
        0 : "&#128515;",
        1 : "&#128552;",
        2 : "&#128544;",
        3 : "&#128542;",
        4 : "&#129314;",
        5 : "&#128556;",
        6 : "&#128547;"
    }

    return emoji2code[result] 
