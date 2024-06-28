from flask import Flask, render_template, request
import pandas as pd
import json
import os
from fuzzywuzzy import process
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

app = Flask(__name__, template_folder='template')

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load route data from CSV
data = pd.read_csv("C://Users//ELCOT//Downloads//csv_data.csv")
unique_locations = pd.concat([data['From'], data['To']]).unique()
unique_locations_df = pd.DataFrame(unique_locations, columns=['Location'])
unique_locations_df['Location'] = unique_locations_df['Location'].str.lower()

# Define the current directory and JSON file path
current_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_dir,'data', 'data.json')

def load_training_data(file_path):
    """Load training data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data['data']
    except FileNotFoundError:
        print(f"Error: The file at {file_path} does not exist.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the JSON file: {e}")
        return None

training_data = load_training_data(json_file_path)

def find_response(user_query):
    """Finds and returns a response based on user query from loaded JSON data."""
    user_query = user_query.lower().strip()
    for item in training_data:
        if user_query == item.get('User', "").lower():
            return item.get('Example', "No specific response found.")
    return "No data available or query was irrelevant."

def find_location_(query):
    """Finds and returns route information based on location from a CSV file."""
    words = query.lower().split()
    location_matches = {word: process.extractOne(word, unique_locations_df['Location'].values, score_cutoff=80) for word in words}
    to_index = words.index('to') if 'to' in words else -1

    from_location, to_location = None, None
    for word, match in location_matches.items():
        if match:
            if to_index != -1 and words.index(word) > to_index:
                to_location = match[0]
            elif to_index == -1 or words.index(word) < to_index:
                from_location = match[0]

    if not from_location or not to_location:
        return "No complete route data found in your query."

    result = data[(data['From'].str.lower() == from_location.lower()) & (data['To'].str.lower() == to_location.lower())]
    if result.empty:
        return f"No route from {from_location} to {to_location} found."
    
    return format_result(result, query)

def format_result(result, query):
    """Formats the result into a string with specified attributes in bold based on the query."""
    columns_to_display = extract_attributes(query, result)
    attribute_values = {col: set() for col in columns_to_display}
    for _, row in result.iterrows():
        for col in columns_to_display:
            attribute_values[col].add(row[col])
    
    formatted_result = ""
    for col, values in attribute_values.items():
        formatted_result += f"<strong>{col}:</strong> " + ", ".join(map(str, values)) + "<br>"
    return formatted_result.strip()

def extract_attributes(query, result):
    """Extracts and returns specific attributes from the result based on the query."""
    query = query.lower()
    patterns = {
        "Depot": r"depots?|central depot|central|depot",
        "Route No.": r"route number|route no\.?|routes?|route",
        "From": r"from|departure points?|departing from|depart",
        "To": r"to|destinations?|arrival points?|arriving at",
        "Route Length": r"route length|length|distance",
        "Type": r"types?|kind|kinds|categories?|class(es)?|buses?|bus type|bus category",
        "No. of Service": r"services?|number of services?|number of buses?|no\.? of buses?|no\.? of services?|available service|services",
        "Departure Timings": r"departure timings?|timings?|departure times?|time|timing",
        "All": r"details|all data|whole"
    }
    attribute_names = []
    query_tokens = nltk.word_tokenize(query)
    for attribute, pattern in patterns.items():
        if re.search(pattern, query):
            attribute_names.append(attribute)

    if "All" in attribute_names or not attribute_names:
        return list(result.columns)
    return attribute_names

@app.route('/', methods=['GET', 'POST'])
def query_form():
    if request.method == 'POST':
        user_query = request.form['query']
        if "route" in user_query or "from" in user_query and "to" in user_query:
            response = find_location_(user_query)
        else:
            response = find_response(user_query)
        return render_template('results.html', response=response)
    return render_template('form.html')

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)

