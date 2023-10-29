import streamlit as st
import folium as fl
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from datetime import datetime
import wikipedia
import pandas as pd
from sklearn.model_selection import train_test_split
from countryinfo import CountryInfo
from folium.plugins import HeatMap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Load data from Excel
df = pd.read_csv('sorted_forest_percentage.csv')


# def get_capital(country):
    
#     country = CountryInfo(country)
#     capital = country.capital() 
#     return capital 
    
# def getloc_lats_longs(location):
    # Accepts: list of location names
    # returns: list of lists containing latitude and longitude coordinates for the specified locations
    
    # geolocator = Nominatim(user_agent="bytescout", timeout=None)
    # loc_lat_long = geolocator.geocode(query = location)

    # return(loc_lat_long.latitude, loc_lat_long.longitude)
    




# df_aug = pd.DataFrame(data, columns=['capital', 'lat', 'long'])
# df.to_csv("forest_aug.csv", index=False)
    





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

def get_country_for_graph(name):
    df_country = df[df['country_name']==name]
    # print(df_country)
    return df_country

get_country_for_graph("India")


geolocator = Nominatim(user_agent="bytescout", timeout=None)



# countries = list(set(list(df['country_name'])))


    


# print(getloc_lats_longs(capitals))



# Add a title
st.title('Deforestation Prediction App with Interactive Map')

# Add a Subheader for the map
st.subheader('Click on the Map to Make a Prediction')

def get_pos(lat,lng):
    return lat,lng


# heat map 

path = "C:/Users/RIBHAV OJHA/Desktop/Uni/Guh23/forest_aug_na_removed.csv"
df_2 = pd.read_csv(path, na_values= '#DIV/0!' )
df_2['forest_area_percentage'] = df_2['forest_area_percentage'].apply(lambda x: float(x))
df_2 = df_2.dropna()

lats_longs_weight = list(map(list, zip(df_2["lat"],
                          df_2["long"], 
                          df_2["forest_area_percentage"]
                         )
               )
           )
    


print(lats_longs_weight)
# print(lats_longs_weight[:5])
m = fl.Map()

m.add_child(fl.LatLngPopup())


for coords in lats_longs_weight:

    # map_obj = fl.Map(location = [38.27312, -98.5821872], zoom_start = 4)

    HeatMap(lats_longs_weight).add_to(m)
    break
map = st_folium(m, height=350, width=700)
# map_2 = st_folium(map_obj, height=350, width=700)



user_year = st.text_input("Enter year to be predicted within the next 5 years")

# # Function to make predictions based on the location
# def make_prediction_on_click(latitude, longitude):
#     # Replace this with your machine learning model
#     # Perform the prediction based on the clicked location
#     prediction = "Deforestation"  # Replace with your actual prediction logic
#     return prediction

# Get the latitude and longitude when the user clicks on the map
if st.button('Click on the Map to Get Prediction'):
    try:
        if not user_year.isdigit():
            st.write("Please enter numerical value")
        else:
            user_year = int(user_year)
            current_year = datetime.now().year
            if (user_year < current_year) or (user_year > (current_year + 5)):
                st.write('Please enter a year within the next 5 years')
            else:
                
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
                percentage1=round(predicted_percentage,2)
                st.write("The predicted forest percentage in {country} in {year} is {per} %".format(country=country, year=user_year, per=percentage1))
     
    except Exception as e:
        st.write(f"Data not found for the specified country. Please try other countries")
    
if st.button('Click on the Map to Get Historical Data'):
    try:
        if not user_year.isdigit():
            st.write("Please enter numerical value")
        else:
            user_year = int(user_year)
            current_year = datetime.now().year
            if (user_year > 2016):
                st.write('Please enter a previous year')
            else:
                
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
                percentage1=round(predicted_percentage,2)
                st.write("The historical forest percentage in {country} in {year} is {per} %".format(country=country, year=user_year, per=percentage1))
                country_data = get_country_for_graph(country)
                import plotly.express as px

                # st.line_chart(data=country_data, x='year', y='forest_area_percentage', use_container_width=True)
                # Create a line chart with sensible year scale
                fig = px.line(country_data, x='year', y='forest_area_percentage', title='Forest Area Percentage Over the Years')
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.write(f"Data not found for the specified country. Please try other countries")
        st.write(e)
