from flask import Flask, jsonify, render_template, request, Blueprint
import google.generativeai as genai
import re
import urllib.parse  # Import urllib.parse for URL encoding

app = Flask(__name__, static_folder='static')

genai.configure(api_key="AIzaSyC6UAKRyYYYH7KLa-nsZTYKNnjnc95k24Y")
model = genai.GenerativeModel('gemini-1.5-flash')

# Recommand File Code
@app.route('/')
def index_ui():
    return render_template('index.html')
@app.route('/submit_ingredients', methods=['POST'])
def submit_ingredients():
    data = request.get_json()
    rn = data.get("prompt")
    
    responses = []

    for i in range(3):  # Generate three dishes
        inp = (
            "Generate a " + rn + 
            " dish with a unique style of making, including the ingredients and the steps to prepare it. "
            "Format: Dish Name: [Name], Ingredients: [List], Steps: [Steps]"
        )
        try:
            response = model.generate_content(inp)  # Use the correct method name
            responses.append(response.text)  # Adjust based on the actual return format
        except Exception as e:
            responses.append(f"Error: {e}")

    return jsonify(responses)


if __name__ == '__main__':
    app.run(debug=True)
