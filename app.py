import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load and prepare data
@st.cache_data
def load_data():
    # Update this path if your dataset.csv is in a different location
    df = pd.read_csv('dataset.csv')
    
    # Select original features (excluding engineered ones and unnamed index)
    original_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = df[original_features]
    y = df['Outcome']
    
    return df, X, y

# Function to train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy, X_test, y_test

# Load data
df, X, y = load_data()

# Train model
model, accuracy, X_test, y_test = train_model(X, y)

# Streamlit UI
st.title("Diabetes Prediction App")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["View Data", "Visualizations", "Model Performance", "Predict Diabetes"])

if page == "View Data":
    st.header("Dataset Overview")
    st.dataframe(df.head())
    st.write(f"Dataset shape: {df.shape}")

elif page == "Visualizations":
    st.header("Data Visualizations")
    
    # Correlation Matrix
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Outcome Distribution
    st.subheader("Outcome Distribution")
    f, ax = plt.subplots(1, 2, figsize=(12, 5))
    df['Outcome'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
    ax[0].set_title('Outcome')
    ax[0].set_ylabel('')
    sns.countplot(x='Outcome', data=df, ax=ax[1])
    ax[1].set_title('Outcome')
    st.pyplot(f)

elif page == "Model Performance":
    st.header("Model Performance")
    st.write(f"Model Accuracy on Test Set: {accuracy:.2f}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

elif page == "Predict Diabetes":
    st.header("Predict Diabetes Outcome")
    
    # Input fields for features
    pregnancies = st.number_input("Pregnancies", min_value=0.0, max_value=20.0, value=0.0)
    glucose = st.number_input("Glucose", min_value=0.0, max_value=200.0, value=100.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=150.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
    insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
    
    if st.button("Predict"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        })
        
        # Predict
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]  # Probability of positive class
        
        if prediction == 1:
            st.error(f"High risk of diabetes (Probability: {prob:.2f})")
        else:
            st.success(f"Low risk of diabetes (Probability: {prob:.2f})")