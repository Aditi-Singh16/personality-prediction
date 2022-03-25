from flask import Flask, request
from pickle import load

app = Flask(__name__)


@app.route("/personalityPrediction", methods=["GET"])
def home():
    args = request.args
    my_vector = load(open("app/vectorizer_pkl", "rb"))
    my_model = load(open("app/model_pkl", "rb"))
    my_post = [args.get("text")]
    print(my_post)
    my_post_trans = my_vector.transform(my_post).toarray()
    res = my_model.predict(my_post_trans)
    return str(res[0])
