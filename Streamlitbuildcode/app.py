# import streamlit as st
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model

# # Function to load the model and scaler
# def load_artifacts():
#     model = load_model('stock_prediction_model.h5')
#     with open('scaler.pkl', 'rb') as file:
#         scaler = pickle.load(file)
#     return model, scaler

# # Function to make predictions
# def predict_stock_price(model, scaler, input_data):
#     # Convert input_data to numpy array and reshape
#     input_data = np.array(input_data).reshape(1, -1)
#     # Scale the input
#     scaled_input = scaler.transform(input_data)
#     # Reshape for the model
#     scaled_input = scaled_input.reshape(1, scaled_input.shape[0], 1)
#     # Make prediction
#     prediction = model.predict(scaled_input)
#     # Inverse transform the prediction
#     original_prediction = scaler.inverse_transform(prediction)
#     return original_prediction[0][0]

# # Streamlit App
# def main():
#     st.title('Stock Price Prediction App')

#     # Loading model and scaler
#     model, scaler = load_artifacts()

#     # User input
#     st.subheader('Enter the stock data:')
#     input_data = st.text_input('Enter data as comma separated values (CSV)')

#     if st.button('Predict'):
#         if input_data:
#             try:
#                 # Processing user input
#                 input_list = [float(i.strip()) for i in input_data.split(',')]
#                 # Predicting stock price
#                 prediction = predict_stock_price(model, scaler, input_list)
#                 st.success(f'Predicted Stock Price: {prediction}')
#             except ValueError:
#                 st.error('Please enter valid numeric values.')
#         else:
#             st.error('Please enter input data.')

# if __name__ == '__main__':
#     main()


import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Function to load the model and scaler
def load_artifacts():
    model = load_model('stock_prediction_model.h5')
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

# Function to make predictions
def predict_stock_price(model, scaler, input_data):
    # Make sure input_data is 100 values
    if len(input_data) != 100:
        raise ValueError("The input data must contain 100 values.")
    # Scale the input
    scaled_input = scaler.transform(np.array(input_data).reshape(-1, 1))
    # Reshape for the model
    scaled_input = scaled_input.reshape(1, 100, 1)
    # Make prediction
    prediction = model.predict(scaled_input)
    # Inverse transform the prediction
    original_prediction = scaler.inverse_transform(prediction)
    return original_prediction[0][0]

# Streamlit App
def main():
    st.title('Stock Price Prediction App')

    # Loading model and scaler
    model, scaler = load_artifacts()

    # User input
    st.subheader('Enter the stock data:')
    input_data = st.text_input('Enter 100 days of stock prices as comma separated values (CSV)')

    if st.button('Predict'):
        if input_data:
            try:
                # Processing user input
                # input_list = [float(i.strip()) for i in input_data.split(',')]
                input_list=input_data.split(',')
                for i in range(len(input_list)):
                    input_list[i]=float(input_list[i])
                # print(input_list)
                # print(len(input_list))
                prediction = predict_stock_price(model, scaler, input_list)
                st.success(f'Predicted Stock Price: {prediction}')
            except ValueError as e:
                st.error(f'Error: {e}')
        else:
            st.error('Please enter input data.')

if __name__ == '__main__':
    main()
