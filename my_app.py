import streamlit as st
import folium as fl
from streamlit_folium import st_folium

# Add a title
st.title('Deforestation Prediction App with Interactive Map')

# Add a Subheader for the map
st.subheader('Click on the Map to Make a Prediction')
def get_pos(lat,lng):
    return lat,lng

m = fl.Map()

m.add_child(fl.LatLngPopup())

map = st_folium(m, height=350, width=700)

# Function to make predictions based on the location
def make_prediction_on_click(latitude, longitude):
    # Replace this with your machine learning model
    # Perform the prediction based on the clicked location
    prediction = "Deforestation"  # Replace with your actual prediction logic
    return prediction

# Get the latitude and longitude when the user clicks on the map
if st.button('Click on the Map to Get Prediction'):
    data = get_pos(map['last_clicked']['lat'],map['last_clicked']['lng'])

    if data is not None:
        st.write(data)

