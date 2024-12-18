#import packages
import streamlit as st
import streamlit.components.v1 as components
import pickle
import pandas as pd
from category_encoders import BinaryEncoder
import sklearn
from sklearn.ensemble import RandomForestClassifier

#encoder class needs to be defined here
class DataEncoder:
    def __init__(self, label_encode_cols):
        self.label_encode_cols = label_encode_cols
        self.label_encoders = {}
        self.binary_encoders = {}
        
    def fit(self, df):
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if col in self.label_encode_cols:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col])
            else:
                self.binary_encoders[col] = BinaryEncoder()
                self.binary_encoders[col].fit(df[col])
        return self
    
    def transform(self, df):
        df_copy = df.copy()
        # Apply Label Encoders
        for col, encoder in self.label_encoders.items():
            df_copy[col] = encoder.transform(df_copy[col]).astype(float)
        # Apply Binary Encoders
        for col, encoder in self.binary_encoders.items():
            encoded = encoder.transform(df_copy[col]).astype(float)
            df_copy = df_copy.join(encoded).drop(columns=[col])
        return df_copy
    
    def fit_transform(self, df):
        return self.fit(df).transform(df)

#page config
st.set_page_config(page_title='Income predictor')

#streamlit frontend
st.title("Income predictor")
age = st.number_input("What is your age?", min_value=0, step=1, format="%d")
workclass = st.selectbox("What is your work class?", ["Federal-gov", "Local-gov", "Never-worked", "Private", "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay"])
education = st.selectbox("What is your education?", ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Some-college", "Prof-school", "Assoc-acdm", "Assoc-voc", "Bachelors", "Masters", "Doctorate"])
maritalStatus = st.selectbox("What is your marital status?", ["Never-married", "Married-civ-spouse", "Married-AF-spouse", "Married-spouse-absent", "Divorced", "Separated", "Widowed"])
occupation = st.selectbox("What is your occupation?", ["Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct", "Other-service", "Priv-house-serv", "Protective-serv", "Sales", "Tech-support", "Transport-moving"])
relationship = st.selectbox("What is your relationship?", ["Not-in-family", "Unmarried", "Husband", "Wife", "Other-relative", "Own-child"])
race = st.selectbox("What is your race?", ["Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "White", "Other"])
sex = st.selectbox("What is your sex?", ["Male", "Female"])
capitalGain = st.selectbox("Did you report capital gains last year?", ["No", "Yes"])
capitalLoss = st.selectbox("Did you report capital losses last year?", ["No", "Yes"])
hours = st.number_input("How many hours per week do you work?", min_value=0, step=1, format="%d")
nativeCountry = st.selectbox("What is your native country?", ["Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic", "Ecuador", "El-Salvador", "England", "France", "Germay", "Greece", "Guatemala", "Haiti", "Holland-Netherlands", "Honduras", "Hong", "Hungary", "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan", "Laos", "Mexico", "Nicaragua", " Outlying-US(Guam-USVI-etc)", "Peru", "Philippines", "Poland", "Portugal", "Puerto-Rico", "Scotland", "South", "Taiwan", "Thailand", "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia"])
model_choice = st.radio("Select the Model", ("Decision Tree", "Random Forest"))

#on button click
if st.button("Predict Salary"):
    #turn the data into a dataframe that matches the raw data before preprocessing
    data = {
    "age": [age],
    "workclass": [workclass],
    "education": [education],
    "marital-status": [maritalStatus],
    "occupation": [occupation],
    "relationship": [relationship],
    "race": [race],
    "sex": [sex],
    "capital-gain": [1 if capitalGain == 'yes' else 0],
    "capital-loss": [1 if capitalLoss == 'yes' else 0],
    "hours-per-week": [hours],
    "native-country": [nativeCountry]
}
    person = pd.DataFrame(data)
    
    #scale numeric columns with scalers from train data preprocessing
    numeric_cols= ["age","hours-per-week"]
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    person[numeric_cols] = scaler.transform(person[numeric_cols])

    #encode categorical columns with encoder from train data preprocessing
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    person = encoder.transform(person)

    #model prediction
    if model_choice == 'Random Forest':
        with open('randomForest.pki', 'rb') as f:
            rf = pickle.load(f)
        model = pickle.load(open("/home/aronconnors/salaryPredictor/randomForest.pki",'rb'))
        result = model.predict(person)
    elif model_choice == 'Decision Tree':
        with open('decisionTree.pki', 'rb') as f:
            rf = pickle.load(f)
        model = pickle.load(open("/home/aronconnors/salaryPredictor/decisionTree.pki",'rb'))
        result = model.predict(person)
    
    #display result
    if result[0] == 0:
        st.header("Low salary, under $55K")
    elif result[0] == 1:
        st.header("High salary, over $55K")
    