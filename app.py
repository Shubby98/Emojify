from flask import Flask , render_template , request
from text2emoji import *
import tensorflow as tf

model , word2idx= load_model()

app = Flask(__name__)

#graph = tf.get_default_graph()

@app.route('/' )
def index():
    data = {
        "pred" : False
    }
    return render_template("home.html" , data = data )

@app.route('/get_emoji' , methods = ['POST' , 'GET'])
def get_emoji():
    global model , word2idx
    text = request.form['inputtxt']
    print(text)
    x = textpreprocessing(text , word2idx)
    result = predict_emoji(x ,model)
    emoji = get_html_emoji(result)
    return text + " " + emoji

@app.route('/update' , methods = ['POST'])
def update():
    global model, word2idx
    text = request.form['inputtxt']
    actual = request.form['actual']
    print(text)
    print(actual)
    actual = get_emoji_num(actual)
    x = textpreprocessing(text , word2idx)
    model = update_model(x, actual , model)

    return "Model is updated your changes will reflect soon"
    
if __name__ == "__main__":
    app.run(debug= True, port= 5000)