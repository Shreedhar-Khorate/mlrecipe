from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai
import os

app = Flask(__name__)

# Configure the Gemini AI model
genai.configure(api_key="AIzaSyC6UAKRyYYYH7KLa-nsZTYKNnjnc95k24Y")  # Replace with your actual Gemini API key
model = genai.GenerativeModel('gemini-1.5-flash')

# Load the dataset
recipe_df = pd.read_csv('tt.csv')

# Handle missing data
recipe_df.fillna('', inplace=True)

# Preprocess Ingredients
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_ingredients = vectorizer.fit_transform(recipe_df['ingredients_list'])

# Normalize Numerical Features
scaler = StandardScaler()
numerical_columns = ['calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium', 'fiber']
X_numerical = scaler.fit_transform(recipe_df[numerical_columns])

# Combine Features
X_combined = np.hstack([X_numerical, X_ingredients.toarray()])

# Train KNN Model
knn = NearestNeighbors(n_neighbors=3, metric='cosine')
knn.fit(X_combined)

# Function to Recommend Recipes
def recommend_recipes(input_features):
    try:
        # Preprocess input features
        input_numerical = pd.DataFrame([input_features[:7]], columns=numerical_columns)  # Ensure feature names match
        input_features_scaled = scaler.transform(input_numerical)  # Transform numerical data
        
        # Transform ingredients
        input_ingredients_transformed = vectorizer.transform([input_features[7]])
        
        # Combine features
        input_combined = np.hstack([input_features_scaled, input_ingredients_transformed.toarray()])
        
        # Get KNN recommendations
        distances, indices = knn.kneighbors(input_combined)
        recommendations = recipe_df.iloc[indices[0]]
        return recommendations[['recipe_name', 'ingredients_list', 'image_url']]
    except Exception as e:
        raise ValueError(f"Error in recommendation: {str(e)}")

@app.route('/', methods=['POST', 'GET'])
def recommend():
    if request.method == 'POST':
        try:
            # Get input values from the form
            protein = float(request.form.get('proteinAmount', 0))
            calories = float(request.form.get('calories', 0))
            fat = float(request.form.get('fat', 0))
            carbohydrates = float(request.form.get('Carbo-Hydrates', 0))
            sodium = float(request.form.get('Sodium', 0))
            fiber = float(request.form.get('Fiber', 0))
            cholesterol = float(request.form.get('Cholesterol', 0))
            ingredients = request.form.get('ingredients', '')

            # Prepare input data for recommendation
            input_data = [calories, fat, carbohydrates, protein, cholesterol, sodium, fiber, ingredients]
            recommendations = recommend_recipes(input_data)
            
            # Extract recipe names and ingredients
            recipe_details = recommendations[['recipe_name', 'ingredients_list', 'image_url']]

            # Initialize an empty list to store steps
            steps = []

            # Loop through each recipe and generate steps
            for _, row in recipe_details.iterrows():
                inp = (
                    f"Generate only step-by-step instructions to prepare '{row['recipe_name']}' "
                    f"using the following ingredients: {row['ingredients_list']}. "
                    "Include estimated quantities and qualities for each ingredient."
                )

                try:
                    # Generate content using the AI model
                    response = model.generate_content(inp)
                    steps.append(response.text)  # Append the generated content
                except Exception as e:
                    steps.append(f"Error generating steps: {str(e)}")  # Handle AI errors gracefully

            # Add the generated steps as a new column in the recommendations DataFrame
            recommendations['steps'] = steps

            # Render the template with the recommendations and steps
            return render_template('index.html', recommendations=recommendations.to_dict(orient='records'))
        
        except Exception as e:
            # Handle errors gracefully
            return render_template('index.html', error=f"Error: {str(e)}", recommendations=[])

    # For GET request, render the page with no recommendations
    return render_template('index.html', recommendations=[])


if __name__ == '__main__':
    app.run(debug=True)
