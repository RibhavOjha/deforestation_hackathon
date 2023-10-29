import streamlit as st
import folium as fl
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data from Excel
df = pd.read_csv('sorted_forest_percentage.csv')

# function to train the model
def train_country(df):
    X = df[["year"]] # Features
    y = df[["forest_area_percentage"]]  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Create the linear regression model
    model = LinearRegression()


    # Train the model using the training data
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model

# function to get prediction from trained model
def predict(model, future_year):
    future_forest_area = model.predict([[future_year]])
    return future_forest_area


# This function is called by the UI
# It returns the prediction for the requested year and country
def get_country_prediction(name, year):
    df_country = df[df['country_name']==name]
    if len(df_country)==0:
        return 0
    # train model for that country
    trained_model = train_country(df_country)
    # get prediction
    prediction = predict(trained_model, year)
    print("predicted forest area is:", prediction)
    return prediction.tolist()[0][0]

# Add a title
st.title('Deforestation Prediction App with Interactive Map')

# Add a Subheader for the map
st.subheader('Click on the Map to Make a Prediction')

def get_pos(lat,lng):
    return lat,lng

m = fl.Map()

m.add_child(fl.LatLngPopup())

map = st_folium(m, height=350, width=700)

user_year = st.text_input("Enter year to be predicted within the next 5 years")

# Function to make predictions based on the location
def make_prediction_on_click(latitude, longitude):
    # Replace this with your machine learning model
    # Perform the prediction based on the clicked location
    prediction = "Deforestation"  # Replace with your actual prediction logic
    return prediction

# Get the latitude and longitude when the user clicks on the map
if st.button('Click on the Map to Get Prediction'):
    if not user_year.isdigit():
        st.write("Please enter numerical value")
    else:
        user_year = int(user_year)
        current_year = datetime.now().year
        if (user_year <= current_year) or (user_year > (current_year + 5)):
            st.write('Please enter a year within the next 5 years')
        else:
            pass
            data = get_pos(map['last_clicked']['lat'],map['last_clicked']['lng'])

            # if data is not None:
            #     st.write(data)
            
            geolocator = Nominatim(user_agent="geoapiExercises")
            coord = data
            location = geolocator.reverse(coord, exactly_one=True, language='en')
            address = location.raw['address']
            country = address.get('country', '')
            st.write(country)
            predicted_percentage = get_country_prediction(country,user_year)
            st.write("The predicted forest percentage in {country} in {year} is {per}".format(country=country, year=user_year, per=predicted_percentage))
