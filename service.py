
# import Orange
import pickle
import numpy as np
from flask import Flask, jsonify, request
import logging
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# define logging level
logging.basicConfig(level=logging.INFO)

# load model created with scilearn
model = pickle.load(open("./models/tree_model.sav", "rb"))
# load the count vectorizer
cv = pickle.load(open("./vectorizers/count_vectorizer.sav","rb"))
# ------------------------------------
# API: we will use Flask to create a simple API
app = Flask(__name__)

# define the route for the API
# http://<host>:<port>/usage
# returns the usage of the API
@app.route("/usage")
def usage():
    return jsonify(
        {
            "text": "Hey. I'm quite pleased with the current political views of the pelican. Pelicans are awesome. They have beautiful beaks."
        }
    )

# define the route which will be used to predict
@app.route("/predict", methods=["POST"])
def predict():
    
    #going to use try catch. Frontend should never break
    try:
        # Get the text from the JSON payload
        data = request.json
        text = data.get('text', '')
        print (text)
        #cv = CountVectorizer(lowercase=True, stop_words='english',ngram_range = (1, 1))        
        
        # load the prediction with the text. no validation is done. 
        logging.info("Starting text processing. Please wait...")
        cvText = cv.transform([text])
        bowText = pd.DataFrame(cvText.toarray(), columns=cv.get_feature_names_out())
        logging.info("Starting prediction. Please wait...")
        predictResult = model.predict(bowText)
        logging.info("Prediction ended.")
        logging.info(f"Prediction: {predictResult[0]}")
        if (predictResult[0] == 1) :
            returnText = "Positive"
        else :
            returnText = "Negative"
            
        # returns the result
        return jsonify(
            {
                "prediction": returnText
            }
        ), 201 # created
    except Exception as e:
        return jsonify({'result': 'error', 'message': str(e)}),400
    


# handles missing routes (404)
@app.errorhandler(404)
def not_found(e):
    return jsonify(
        {
            "code": e.code,
            "error": str(e)
        }
    ), 404


if __name__ == "__main__":
    # run the API
    app.run(
        host='0.0.0.0',  # needed to access from outside the container. #Nuno This is actually a loopback endereço. Gonna leave it like this
        port=5001,  # define the port
        debug=True # e.g., restarts the API when the code changes. This is awesome. JIT. Didn't know it existed!
    )

