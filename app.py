from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

@lru_cache(maxsize=None)
def load_models():
    model = tf.keras.models.load_model('model/tourism_classifier.h5')
    tfidf_vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')
    tourism_df = pd.read_csv('data/tourism_with_id.csv')
    return model, tfidf_vectorizer, label_encoder, tourism_df

def prepare_place_features(tourism_df):
    tourism_df['text_features'] = tourism_df.apply(
        lambda x: f"{x['Place_Name']} {x['Description']} {x['City']}", axis=1
    )
    return tourism_df

def predict_category(model, tfidf_vectorizer, label_encoder, text):
    input_vector = tfidf_vectorizer.transform([text]).toarray()
    prediction = model.predict(input_vector)
    
    probabilities = prediction[0]
    predicted_category_index = np.argmax(probabilities)
    predicted_category = label_encoder.inverse_transform([predicted_category_index])[0]
    
    categories = label_encoder.classes_
    category_probs = {cat: float(prob) for cat, prob in zip(categories, probabilities)}
    
    return predicted_category, category_probs, probabilities

def calculate_neural_network_similarity(input_probs, all_places_probs):
    return cosine_similarity(input_probs.reshape(1, -1), all_places_probs)[0]

def get_recommendations_nn(place_name, tourism_df, model, tfidf_vectorizer, label_encoder, n_recommendations=15):
    # Prepare place features
    df = prepare_place_features(tourism_df.copy())
    
    # Get input place text features
    input_place = df[df['Place_Name'].str.lower() == place_name.lower()]
    if input_place.empty:
        input_text = place_name
    else:
        input_text = input_place['text_features'].iloc[0]
    
    # Predict category for input place
    predicted_category, category_probs, input_probs = predict_category(
        model=model,
        tfidf_vectorizer=tfidf_vectorizer,
        label_encoder=label_encoder,
        text=input_text
    )
    
    # Pre-calculate all predictions for efficiency
    all_vectors = tfidf_vectorizer.transform(df['text_features']).toarray()
    all_predictions = model.predict(all_vectors)
    
    # Calculate neural network similarities
    nn_similarities = calculate_neural_network_similarity(input_probs, all_predictions)
    df['nn_similarity'] = nn_similarities
    
    if not input_place.empty:
        df = df[df['Place_Name'] != place_name]
    
    # Sort recommendations based on neural network similarity
    recommendations = df.nlargest(n_recommendations, 'nn_similarity')
    detailed_recommendations = []
    
    for _, row in recommendations.iterrows():
        similarity_details = {
            'name': row['Place_Name'],
            'city': row['City'],
            'category': row['Category'],
            'price': f"Rp {row['Price']:,}",
            'description': row['Description'][:200] + '...' if len(row['Description']) > 200 else row['Description'],
            'nn_similarity_score': float(row['nn_similarity']),
            'explanation': f"Tempat ini direkomendasikan karena memiliki skor kemiripan neural network sebesar {row['nn_similarity']:.2f}."
        }
        detailed_recommendations.append(similarity_details)
    
    return detailed_recommendations, predicted_category, category_probs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request content type must be application/json.'
            }), 415
        
        data = request.get_json()
        if 'place_name' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "place_name" in the request body.'
            }), 400
        
        place_name = data['place_name']
        model, tfidf_vectorizer, label_encoder, tourism_df = load_models()
        
        recommendations, category, category_probs = get_recommendations_nn(
            place_name=place_name,
            tourism_df=tourism_df,
            model=model,
            tfidf_vectorizer=tfidf_vectorizer,
            label_encoder=label_encoder
        )
        
        return jsonify({
            'success': True,
            'predicted_category': category,
            'category_probabilities': category_probs,
            'recommendations': recommendations
        })
    
    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)

