from flask import Flask , render_template , request
from text2emoji import *

model , word2idx = load_model()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/get_emoji' , methods = ['POST' , 'GET'])
def get_emoji():
    text = request.form['text']
    x = textpreprocessing(text , word2idx)
    result = predict_emoji(x ,model)
    emoji = get_html_emoji(result)
    return text + emoji

if __name__ == "__main__":
    app.run(debug= True)