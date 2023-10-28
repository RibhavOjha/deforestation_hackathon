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





# Create a map using st.map
# Replace the default coordinates with the location of interest
# For example, you can use the coordinates of a forest area
# Here, we use a sample location in the Amazon Rainforest
# Create a map using st.map
# Replace the default coordinates with the location of interest
# For example, you can use the coordinates of a forest area
# Here, we use a sample location in the Amazon Rainforest
# map_data = [{'latitude': -3.4653, 'longitude': -62.2159}]
# st.map(map_data)



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
    # click_data = st.pydeck_chart()
    # click_data = st.session_state.pydeck_viewport
    # latitude = click_data['latitude']
    # longitude = click_data['longitude']
    # prediction = make_prediction_on_click(latitude, longitude)
    # st.write(f'Prediction for location {latitude}, {longitude}: {prediction}')
