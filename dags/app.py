import streamlit as st
import numpy as np
import pickle

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_quantity(model, input_data):
    return model.predict(input_data)

def main():
    st.title('Crop Yield Prediction')

    # Sidebar input controls
    st.sidebar.header('Input Parameters')
    N = st.sidebar.number_input('Nitrogen (N)', min_value=0.0, max_value=1000.0, value=100.0)
    P = st.sidebar.number_input('Phosphorus (P)', min_value=0.0, max_value=1000.0, value=50.0)
    K = st.sidebar.number_input('Potassium (K)', min_value=0.0, max_value=1000.0, value=75.0)
    temperature = st.sidebar.number_input('Temperature (°C)', min_value=-20.0, max_value=50.0, value=25.0)
    humidity = st.sidebar.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=60.0)
    ph = st.sidebar.number_input('pH', min_value=0.0, max_value=14.0, value=7.0)
    rainfall = st.sidebar.number_input('Rainfall (mm)', min_value=0.0, max_value=1000.0, value=100.0)

    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Load pre-trained model
    model_path = 'C:/Users/PcPack/hha/trained_models/trained_model.pkl'
    model = load_model(model_path)

    if st.button('Predict'):
        try:
            # Make prediction
            prediction = predict_quantity(model, input_data)

            # Display prediction
            st.success(f'Predicted Yield: {prediction[0]}')
        except Exception as e:
            st.error(f'Error: {e}')

if __name__ == '__main__':
    main()
