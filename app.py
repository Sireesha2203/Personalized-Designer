from flask import Flask, render_template, request, redirect, url_for
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='templates')

# Step 1: Database Setup
USER_DB_FILE = 'user_database.pkl'
if not os.path.exists(USER_DB_FILE):
    user_database = {}
    with open(USER_DB_FILE, 'wb') as file:
        pickle.dump(user_database, file)

# Step 4: Initial Model Training
# Load your dataset
dataset_path = 'C:/Users/99896/Desktop/designer/dataset/styles.csv'
df = pd.read_csv(dataset_path)

# Preprocess the dataset (you may need to customize this based on your dataset)
df['description'] = df[['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage', 'productDisplayName']].astype(str).agg(' '.join, axis=1)

# Step 4: Initial Model Training
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['description'])

model = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine')
model.fit(X)

# Save the trained model and vectorizer
model_path = 'trained_model.pkl'
vectorizer_path = 'vectorizer.pkl'

with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(vectorizer_path, 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Step 5: Design Generation
def generate_design(description):
    # Preprocess the input description
    input_description = vectorizer.transform([description])

    # Find the nearest neighbor in the dataset
    _, indices = model.kneighbors(input_description)
    nearest_neighbor_index = indices[0][0]

    # Get the corresponding image ID
    image_id = df.loc[nearest_neighbor_index, 'id']

    # Display the image
    image_path = os.path.join('C:/Users/99896/Desktop/designer/dataset/images', f'{image_id}.jpg')
    image = Image.open(image_path)
    plt.imshow(np.array(image))
    plt.axis('off')
    plt.show()

    return f"Generated Design based on description: {description}"

# Step 7: Update User History
def update_user_history(username, design):
    with open(USER_DB_FILE, 'rb') as file:
        user_database = pickle.load(file)

    if username in user_database:
        user_database[username]['design_history'].append(design)
    else:
        user_database[username] = {'design_history': [design]}

    with open(USER_DB_FILE, 'wb') as file:
        pickle.dump(user_database, file)

# Step 2: User Authentication
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    # Implement actual authentication logic here

    return redirect(url_for('designer', username=username))

# Step 3: Data Collection
@app.route('/designer/<username>', methods=['GET', 'POST'])
def designer(username):
    if request.method == 'POST':
        description = request.form['description']

        # Step 5: Design Generation
        design = generate_design(description)
        update_user_history(username, design)

        return render_template('design.html', username=username, design=design)

    return render_template('designer.html', username=username)

# Step 6: Displaying the Design
@app.route('/history/<username>')
def design_history(username):
    with open(USER_DB_FILE, 'rb') as file:
        user_database = pickle.load(file)

    designs = user_database.get(username, {}).get('design_history', [])

    return render_template('history.html', username=username, designs=designs)

if __name__ == '__main__':
    app.run(debug=True)
