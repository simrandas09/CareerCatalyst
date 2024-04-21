import os
# Activate the virtual environment
os.system("source ./venv/bin/activate")
import tkinter as tk
from PIL import Image, ImageTk
import json
import numpy as np
import nltk
import joblib
import pandas as pd

# Initialize NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Load intents file
with open('/home/simran-das/Career-Catalyst-/PythonScript/intents.json') as file:
    intents = json.load(file)

# Load trained model and transformer
classifier = joblib.load("/home/simran-das/Career-Catalyst-/PythonScript/ml_model.pkl")
transformer = joblib.load("/home/simran-das/Career-Catalyst-/PythonScript/column_transformer.pkl")

# Tkinter GUI initialization
root = tk.Tk()
root.title("Chatbot")

# Set background image
background_image = Image.open("/home/simran-das/Career-Catalyst-/PythonScript/chatbot_background.jpeg")
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_photo)
background_label.place(relwidth=1, relheight=1)

# Function to clean up sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [word.lower() for word in sentence_words if word.isalnum()]
    return sentence_words

# Function to process input and predict career
def process_input_and_predict(input_data):
    # Define the columns expected by the model
    expected_columns = [
        "Acedamic percentage in Operating Systems",
        "percentage in Algorithms",
        "Percentage in Programming Concepts",
        "Percentage in Software Engineering",
        "Percentage in Computer Networks",
        "Percentage in Electronics Subjects",
        "Percentage in Computer Architecture",
        "Percentage in Mathematics",
        "Percentage in Communication skills",
        "Hours working per day",
        "Logical quotient rating",
        "hackathons",
        "coding skills rating",
        "public speaking points",
        "can work long time before system?",
        "self-learning capability?",
        "Extra-courses did",
        "certifications",
        "workshops",
        "talenttests taken?",
        "olympiads",
        "reading and writing skills",
        "memory capability score",
        "Interested subjects",
        "interested career area ",
        "Job/Higher Studies?",
        "Type of company want to settle in?",
        "Taken inputs from seniors or elders",
        "interested in games",
        "Interested Type of Books",
        "Salary Range Expected",
        "In a Realtionship?",
        "Gentle or Tuff behaviour?",
        "Management or Technical",
        "Salary/work",
        "hard/smart worker",
        "worked in teams ever?",
        "Introvert"
    ]

    # Initialize input values with None
    input_values = {col: None for col in expected_columns}

    # Update input values with data from input_data
    for word in input_data:
        if word in input_values:
            input_values[word] = input_data[word]

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_values])

    # Handle missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = np.nan

    return input_df

# Function to predict class
def predict_class(input_data):
    # Clean up the sentence
    sentence_words = clean_up_sentence(input_data)

    # Convert the sentence words to a dictionary
    input_data = {word: None for word in sentence_words}

    # Process input data and predict career
    input_df = process_input_and_predict(input_data)

    # Transform the input data using the saved ColumnTransformer
    transformed_input = transformer.transform(input_df)

    # Make predictions using the trained model
    prediction = classifier.predict(transformed_input)

    # Retrieve response based on predicted class tag
    for intent in intents:  # Iterate over the list of intents
        if 'tag' in intent and intent['tag'] == prediction:
            return intent['responses'][0]

    # If no matching intent is found, return default response
    return "A suitable career for you is {}".format(prediction[0])

# Function to handle user input and display response
def send_message():
    # Get the user input from the input box
    user_input = input_box.get()

    # Predict the class based on the user input
    prediction = predict_class(user_input)

    # Display the response
    display_response(prediction)

# Function to display chatbot response
def display_response(response):
    response_text.config(state=tk.NORMAL)
    response_text.delete(1.0, tk.END)
    response_text.insert(tk.END, response)
    response_text.config(state=tk.DISABLED)

# Input text box
input_box = tk.Entry(root, width=50)
input_box.pack(padx=10, pady=10)

# Send button
send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(pady=10)

# Response display area
response_text = tk.Text(root, height=10, width=50)
response_text.config(state=tk.DISABLED)
response_text.pack(padx=10, pady=10)

# Start the GUI event loop
root.mainloop()