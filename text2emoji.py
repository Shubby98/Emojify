import keras
from nltk import word_tokenize
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences

tokenizer = pickle.load(open('tokenizer.pickle','rb'))
maxlen = 50

def load_model():
    model =  keras.models.load_model("emoji_model.h5")
    pickle_in = open("word2index.pickle","rb")
    word2idx = pickle.load(pickle_in)
    return model , word2idx

def textpreprocessing(text,word2idx):
    text = word_tokenize(text.lower())
    test_sent = tokenizer.texts_to_sequences([text])
    test_sent = pad_sequences(test_sent, maxlen = maxlen)
    
    return test_sent

def predict_emoji(X , model):
    res = model.predict(X)
    res = np.argmax(res)   
    return int(res)
    
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

def update_model(x ,  actual , model):
    new_test = np.vstack([x]*5)
    actual_output = np.zeros((1,7))
    actual_output[0,actual] = 1
    actual_output = np.vstack([actual_output]*5)
    model.compile(loss='categorical_crossentropy' , optimizer = 'adam' , metrics = ['acc'])
    model.fit(new_test , actual_output , epochs = 10 , batch_size = 32 , shuffle = True)
    model.save('emoji_model.h5')
    return model

def get_emoji_num(emoji):
    emoji2num = {
        "happy" : 0,
        "fear" : 1,
        "anger" : 2,
        "sadness" : 3,
        "disgust" : 4,
        "shame" : 5,
        "guilt" : 6
    }
    return emoji2num[emoji]