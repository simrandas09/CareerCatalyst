# Importing necessary packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.metrics import accuracy_score
from sklearn import tree

# Loading data file
data=pd.read_csv('/Users/lagnikadagur/Desktop/chatbot/chatbot/career_pred1.csv')

# Extracting independent and dependent variables
data.dropna(inplace=True)
x=data.iloc[:, :-1]
y=data.iloc[:, -1]

# Splitting data into train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.4)

# Turning categories into numbers
categorical_features=['can work long time before system?', 'self-learning capability?', 'Extra-courses did', 'certifications', 'workshops', 'talenttests taken?', 'olympiads', 'reading and writing skills', 'memory capability score', 'Interested subjects','interested career area ', 'Job/Higher Studies?', 'Type of company want to settle in?', 'Taken inputs from seniors or elders', 'interested in games', 'Interested Type of Books', 'Salary Range Expected', 'In a Realtionship?', 'Gentle or Tuff behaviour?', 'Management or Technical', 'Salary/work', 'hard/smart worker', 'worked in teams ever?', 'Introvert']
one_hot = OneHotEncoder(handle_unknown="ignore")
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")

x_train_transformed = transformer.fit_transform(x_train)
x_test_transformed = transformer.transform(x_test)

# Fitting decision tree classifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_train_transformed, y_train)

# Predicting result
y_pred = classifier.predict(x_test_transformed)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

joblib.dump(classifier,"ml_model.pkl")
joblib.dump(transformer,"column_transformer.pkl")

# Function to predict career
def predict_career(academic_data):
    # Load the trained model and transformer
    classifier = joblib.load("ml_model.pkl")
    transformer = joblib.load("column_transformer.pkl")
    
    # Preprocess the input data
    processed_data = transformer.transform(pd.DataFrame([academic_data]))

    # Make predictions using the trained model
    prediction = classifier.predict(processed_data)

    return prediction