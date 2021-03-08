from flask import Flask, render_template, request
from model.functions import predict, fillmask
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', placeholder = "What's on your mind?")

@app.route('/success', methods=["GET", "POST"])
def success():
    data = request.form['inputTweet']

    if len(data) < 2:
        return render_template('index.html', placeholder = "Please, express yourself")

    # VAR TO BE SENT TO SUCCESS
    else:
        tweet = data
        score = predict(data)
        output = fillmask(data)
        word = output[0]
        recos = output[1]
        return render_template('success-js.html', score = score, tweet = tweet,
        recos = recos, word = word)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port = port)
