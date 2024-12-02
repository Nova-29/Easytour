# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials, firestore
import joblib
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import uuid

app = Flask(__name__)

cred = credentials.Certificate('config/capstone-project-441906-179ceddf7d34.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
users_collection = db.collection('users')

API_KEY = 'AIzaSyA7YpgX2ISV3iHdJvGvaCEoW8WTbNYD0Cw'

def hash_password(password):
    """
    Hashes a password using SHA256.
    """
    return hashlib.sha256(password.encode()).hexdigest()

@lru_cache(maxsize=None)
def load_models():
    model = tf.keras.models.load_model('model/tourism_classifier.h5')
    tfidf_vectorizer = joblib.load('model/tfidf_vectorizer.joblib')
    scaler = joblib.load('model/scaler.joblib')
    label_encoder = joblib.load('model/label_encoder.joblib')
    tourism_df = pd.read_csv('data/tourism_with_id.csv')
    return model, tfidf_vectorizer, scaler, label_encoder, tourism_df

def prepare_place_features(tourism_df):
    """
    Menyiapkan fitur-fitur untuk perhitungan similarity
    """
    tourism_df['text_features'] = tourism_df.apply(
        lambda x: f"{x['Place_Name']} {x['Description']} {x['City']}", axis=1
    )
    return tourism_df

def predict_category(model, tfidf_vectorizer, scaler, label_encoder, place_name):
    """
    Memprediksi kategori tempat wisata
    """
    input_vector = tfidf_vectorizer.transform([place_name]).toarray()
    input_scaled = scaler.transform(input_vector)
    prediction = model.predict(input_scaled)
    
    probabilities = prediction[0]
    predicted_category_index = np.argmax(probabilities)
    predicted_category = label_encoder.inverse_transform([predicted_category_index])[0]
    
    categories = label_encoder.classes_
    category_probs = {cat: float(prob) for cat, prob in zip(categories, probabilities)}
    
    return predicted_category, category_probs

def _generate_similarity_explanation(place, predicted_category):
    """
    Alasan mengapa tempat wisata di rekomendasikan
    """
    reasons = []
    
    if place['similarity_score'] > 0.5:
        reasons.append("memiliki deskripsi dan karakteristik yang sangat mirip")
    elif place['similarity_score'] > 0.3:
        reasons.append("memiliki beberapa kesamaan dalam deskripsi")
    
    if place['Category'] == predicted_category:
        reasons.append(f"termasuk dalam kategori yang sama ({predicted_category})")
    
    if not reasons:
        reasons.append("memiliki beberapa kemiripan umum")
    
    return " dan ".join(reasons)

def get_recommendations(place_name, tourism_df, model, tfidf_vectorizer, scaler, label_encoder, n_recommendations=15):
    """
    Mendapatkan rekomendasi tempat wisata berdasarkan similarity konten
    """
    # Prepare features
    df = prepare_place_features(tourism_df.copy())
    
    # Get predicted category
    predicted_category, category_probs = predict_category(
        model=model,
        tfidf_vectorizer=tfidf_vectorizer,
        scaler=scaler,
        label_encoder=label_encoder,
        place_name=place_name
    )
    
    # Find input place
    input_place = df[df['Place_Name'].str.lower() == place_name.lower()]
    if input_place.empty:
        input_text = place_name
    else:
        input_text = input_place['text_features'].iloc[0]
    
    # Calculate TF-IDF for all places
    content_vectorizer = TfidfVectorizer()
    tfidf_matrix = content_vectorizer.fit_transform(df['text_features'])
    
    # Calculate similarity scores
    if input_place.empty:
        input_vector = content_vectorizer.transform([input_text])
        similarity_scores = cosine_similarity(input_vector, tfidf_matrix)[0]
    else:
        input_idx = input_place.index[0]
        similarity_scores = cosine_similarity(tfidf_matrix[input_idx:input_idx+1], tfidf_matrix)[0]
    
    # Add similarity scores to dataframe
    df['similarity_score'] = similarity_scores
    
    # Calculate category bonus
    df['category_bonus'] = df['Category'].apply(
        lambda x: 0.3 if x == predicted_category else 0
    )
    
    # Calculate final score
    df['final_score'] = df['similarity_score'] + df['category_bonus']
    
    # Filter out input place if it exists
    if not input_place.empty:
        df = df[df['Place_Name'] != place_name]
    
    # Sort by final score and get recommendations
    recommendations = df.nlargest(n_recommendations, 'final_score')
    
    # Prepare detailed recommendations
    detailed_recommendations = []
    for _, row in recommendations.iterrows():
        similarity_details = {
            'name': row['Place_Name'],
            'city': row['City'],
            'category': row['Category'],
            'price': f"Rp {row['Price']:,}",
            'description': row['Description'][:200] + '...' if len(row['Description']) > 200 else row['Description'],
            'similarity_score': float(row['similarity_score']),
            'category_match': row['Category'] == predicted_category,
            'final_score': float(row['final_score']),
            'explanation': _generate_similarity_explanation(row, predicted_category)
        }
        detailed_recommendations.append(similarity_details)
    
    return detailed_recommendations, predicted_category, category_probs

def get_place_image_url(place_name):
    """
    Fetches the image URL of a place using the Google Places API.
    """
    try:
        # Step 1: Find Place from Text
        search_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
        search_params = {
            "input": place_name,
            "inputtype": "textquery",
            "fields": "place_id",
            "key": API_KEY
        }
        search_response = requests.get(search_url, params=search_params)
        search_response.raise_for_status()  # Raise HTTPError for bad responses
        search_data = search_response.json()

        if not search_data.get('candidates'):
            return None

        place_id = search_data['candidates'][0]['place_id']

        # Step 2: Get Place Details
        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_params = {
            "place_id": place_id,
            "fields": "photos",
            "key": API_KEY
        }
        details_response = requests.get(details_url, params=details_params)
        details_response.raise_for_status()
        details_data = details_response.json()

        photos = details_data.get('result', {}).get('photos', [])
        if not photos:
            return None

        photo_reference = photos[0]['photo_reference']

        # Step 3: Build Photo URL
        photo_url = f"https://maps.googleapis.com/maps/api/place/photo"
        return f"{photo_url}?maxwidth=400&photo_reference={photo_reference}&key={API_KEY}"
    
    except requests.exceptions.RequestException as e:
        print(f"Request error for {place_name}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error fetching image URL for {place_name}: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        place_name = data.get('place_name')
        model, tfidf_vectorizer, scaler, label_encoder, tourism_df = load_models()
        
        recommendations, category, category_probs = get_recommendations(
            place_name=place_name,
            tourism_df=tourism_df,
            model=model,
            tfidf_vectorizer=tfidf_vectorizer,
            scaler=scaler,
            label_encoder=label_encoder
        )

        # Add image URLs and assign unique IDs to each recommendation
        # for idx, rec in enumerate(recommendations, start=1):  # ID starts at 1
        #     rec['id'] = idx
        #     rec['image_url'] = get_place_image_url(rec['name'])

        if recommendations:
            first_image_url = get_place_image_url(recommendations[0]['name'])
        else:
            first_image_url = None  # Fallback if no recommendations are provided

        # Assign the same image URL to all recommendations
        for idx, rec in enumerate(recommendations, start=1):  # ID starts at 1
            rec['id'] = idx
            rec['image_url'] = first_image_url

        return jsonify({
            'error': False,
            'code': 200,
            'data': {
                'category_probabilities': category_probs,
                'predicted_category': category,
                'recommendations': recommendations
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/register', methods=['POST'])
def register():
    # Parse the incoming JSON request
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    # Input validation
    if not username or not email or not password:
        return {
            "error": True,
            "message": "Username, email, and password are required."
        }, 400

    # References to the Firestore documents
    user_ref = db.collection('users').document(username)
    counter_ref = db.collection('metadata').document('user_counter')

    try:
        # Check if the user already exists
        if user_ref.get().exists:
            return {
                "error": True,
                "message": f"User {username} already exists."
            }, 400

        # Increment the userID counter
        counter_snapshot = counter_ref.get()
        if counter_snapshot.exists:
            current_id = counter_snapshot.to_dict().get('latest_id', 0)
            new_id = current_id + 1
        else:
            new_id = 1  # Start at 1 if the counter doesn't exist

        # Update the counter document
        counter_ref.set({'latest_id': new_id}, merge=True)

        # Create a new user document
        user_ref.set({
            "userID": new_id,
            "username": username,
            "email": email,
            "password": password,
            "created_at": firestore.SERVER_TIMESTAMP
        })

        return {
            "error": False,
            "message": f"User {username} registered successfully!",
            "userID": str(new_id),
            "email": email,
            "username": username
        }, 201

    except Exception as e:
        return {
            "error": True,
            "message": f"An error occurred: {str(e)}"
        }, 500

@app.route('/login', methods=['POST'])
def login():
    try:
        # Get the data from the request
        data = request.json
        email = data.get('email')
        password = data.get('password')

        # Validate the inputs
        if not email or not password:
            return jsonify({"error": True, "message": "Both email and password are required."}), 400

        # Check if the email exists
        user_query = users_collection.where('email', '==', email).get()
        if not user_query:
            return jsonify({"error": True, "message": "Invalid email or password."}), 401

        # Verify password
        user_data = user_query[0].to_dict()  # Assume email is unique and fetch the first match
        if user_data['password'] != password:
            return jsonify({"error": True, "message": "Invalid email or password."}), 401

        return jsonify({
            "error": False,
            "message": "Login successful.",
            "userID": str(user_data["userID"]),
            "username": user_data["username"],
            "email": email
        }), 200
    except Exception as e:
        return jsonify({"error": True, "message": str(e)}), 500

@app.route('/categories', methods=['POST'])
def update_categories():
    try:
        # Get the data from the request
        data = request.json
        userID = int(data.get('userID'))  # userID used to locate the user's document
        category_preferences = {
            "TamanHiburan": data.get('TamanHiburan', False),
            "Budaya": data.get('Budaya', False),
            "Bahari": data.get('Bahari', False),
            "CagarAlam": data.get('CagarAlam', False),
            "PusatPerbelanjaan": data.get('PusatPerbelanjaan', False),
            "TempatIbadah": data.get('TempatIbadah', False)
        }

        # Validate userID
        if not isinstance(userID, int):
            return jsonify({"error": True, "message": "userID must be an integer."}), 400

        # Validate category preferences
        if not all(isinstance(value, bool) for value in category_preferences.values()):
            return jsonify({"error": True, "message": "All category values must be Boolean."}), 400

        # Check if the userID exists in Firestore
        user_query = users_collection.where('userID', '==', userID).get()
        if not user_query:
            return jsonify({"error": True, "message": "User not found."}), 404

        # Assume the first document in the query result is the correct one
        user_ref = user_query[0].reference

        # Update the user's preferences in Firestore
        user_ref.update({"category_preferences": category_preferences})

        return jsonify({
            "error": False,
            "message": "Category preferences updated successfully.",
            "category_preferences": category_preferences
        }), 200

    except Exception as e:
        return jsonify({"error": True, "message": str(e)}), 500

@app.route('/home', methods=['POST'])
def home_recommendations():
    try:
        data = request.json
        userID = int(data.get('userID'))

        # Validate userID
        if not isinstance(userID, int):
            return jsonify({"error": True, "message": "userID must be an integer."}), 400

        # Check if the userID exists in Firestore
        user_query = users_collection.where('userID', '==', userID).get()
        if not user_query:
            return jsonify({"error": True, "message": "User not found."}), 404

        user_doc = user_query[0].to_dict()
        preferences = user_doc.get("category_preferences", {})

        category_map = {
            "TamanHiburan": "Taman Hiburan",
            "Budaya": "Budaya",
            "Bahari": "Bahari",
            "CagarAlam": "Cagar Alam",
            "PusatPerbelanjaan": "Pusat Perbelanjaan",
            "TempatIbadah": "Tempat Ibadah"
        }
        selected_categories = [
            category_map[key] for key, value in preferences.items() if value
        ]

        if not selected_categories:
            return jsonify({"error": False, "code": 200, "data": []})

        # Load the dataset
        tourism_df = pd.read_csv('data/tourism_with_id.csv')
        ratings_df = pd.read_csv('data/tourism_rating.csv')

        # Merge ratings with the tourism dataset
        tourism_df = pd.merge(tourism_df, ratings_df, on='Place_Id', how='left')
        tourism_df['Rating'] = tourism_df['Rating'].fillna('No rating')  # Handle missing ratings

        # Filter rows where the Category column matches selected categories
        filtered_data = tourism_df[tourism_df['Category'].isin(selected_categories)]

        # Group by category and randomly pick one place from each category
        grouped_recommendations = filtered_data.groupby('Category').apply(
            lambda x: x.sample(1) if len(x) > 0 else None
        ).reset_index(drop=True)

        # Collect recommendations for at least one of each selected category
        recommendations = []
        for _, row in grouped_recommendations.iterrows():
            recommendation = {
                "category": row["Category"],
                "category_match": True,
                "city": row["City"],
                "description": row["Description"][:200] + '...' if len(row["Description"]) > 200 else row["Description"],
                "explanation": f"termasuk dalam kategori yang sama ({row['Category']})",
                "name": row["Place_Name"],
                "price": f"Rp {row['Price']:,}" if pd.notnull(row["Price"]) else "Unknown",
                # "image_url": get_place_image_url(row["Place_Name"]),
                "rating": float(row["Rating"])
            }
            recommendations.append(recommendation)

        # Add additional random places from filtered data until the total is 5
        remaining_places = filtered_data[
            ~filtered_data['Place_Name'].isin(grouped_recommendations['Place_Name'])
        ]
        if not remaining_places.empty:
            additional_recommendations = remaining_places.sample(
                min(50 - len(recommendations), len(remaining_places))
            )
            for _, row in additional_recommendations.iterrows():
                recommendation = {
                    "category": row["Category"],
                    "category_match": True,
                    "city": row["City"],
                    "description": row["Description"][:200] + '...' if len(row["Description"]) > 200 else row["Description"],
                    "explanation": f"termasuk dalam kategori yang sama ({row['Category']})",
                    "name": row["Place_Name"],
                    "price": f"Rp {row['Price']:,}" if pd.notnull(row["Price"]) else "Unknown",
                    # "image_url": get_place_image_url(row["Place_Name"]),
                    "rating": float(row["Rating"])
                }
                recommendations.append(recommendation)

        if recommendations:
            first_image_url = get_place_image_url(recommendations[0]['name'])
        else:
            first_image_url = None  # Fallback if no recommendations are provided

        for idx, recommendation in enumerate(recommendations, start=1):
            recommendation["ID"] = idx
            recommendation["image_url"] = first_image_url

        return jsonify({"error": False, "code": 200, "data": recommendations[:50]})

    except Exception as e:
        return jsonify({"error": True, "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
