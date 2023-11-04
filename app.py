import logging
from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = joblib.load('trained_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Assurez-vous d'avoir sauvegardé le vecteur TF-IDF lors de l'entraînement

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the description from the HTTP request
        data = request.json['description']

        # Log the received data for debugging
        logging.info(f'Received data: {data}')

        # Preprocess the input using the loaded TF-IDF vectorizer
        preprocessed_data = preprocess_data(data)  # Apply preprocessing

        if preprocessed_data is not None:
            # Log the preprocessed data for debugging
            logging.info(f'Preprocessed data: {preprocessed_data}')

            # Make a prediction using the loaded model
            prediction = model.predict([preprocessed_data])

            # Log the prediction for debugging
            logging.info(f'Prediction: {prediction[0]}')

            # Return the prediction as JSON
            return jsonify({'prediction': prediction[0]})

        else:
            return jsonify({'error': 'Failed to preprocess data'})

    except KeyError as e:
        # Handle missing 'description' key
        return jsonify({'error': 'Missing or incorrect input format: ' + str(e)})
    except Exception as e:
        # Handle other exceptions
        logging.error(str(e))
        return jsonify({'error': 'An error occurred. Please check your input data and preprocessing.'})
    
def preprocess_data(data):
    try:
        # Use the same TF-IDF vectorizer as during training
        preprocessed_data = tfidf_vectorizer.transform([data])

        # Convert the sparse matrix to an array
        preprocessed_data = preprocessed_data.toarray()

        return preprocessed_data[0]  # Return the vectorized data as a 1D array
    except Exception as e:
        # Handle exceptions if any
        logging.error(str(e))
        return None



if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)
