from flask import Flask,request,jsonify
from flask_cors import CORS

from bert import Ner

import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

app = Flask(__name__)
CORS(app)

model = Ner("out_base")

@app.route("/predict",methods=['POST'])
def predict():
    text = request.json["text"]
    sentences = sent_detector.tokenize(text)

    try:
        ner_tagged_sentences = []
        for sentence in sentences:
            tagged_sentence = model.predict(sentence)
            ner_tagged_sentences.extend(tagged_sentence)
        # out = ''.join(ner_tagged_sentences)
        return jsonify({"result":ner_tagged_sentences})
    except Exception as e:
        print(e)
        return jsonify({"result":"Model Failed"})

if __name__ == "__main__":
    app.run('0.0.0.0',port=8000)
