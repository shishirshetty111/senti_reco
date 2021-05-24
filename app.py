from flask import Flask, request, render_template
from model import best_5

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if (request.method == "POST"):
        user_input = [x for x in request.form.values()][0]
        # final filtered top5 recommendation
        output=best_5(user_input)
        return render_template('index.html', prediction_text='Top 5 Recommendation', my_list=output)
    else :
        return render_template('index.html')

if __name__ == "__main__":
    app.run()
