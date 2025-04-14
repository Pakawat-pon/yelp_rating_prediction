import streamlit as st
import pandas as pd
import pickle

# Load model
with open("stars_predictor.pkl", "rb") as f:
    model = pickle.load(f)

# Load data just for dropdown values (city, category)
@st.cache_data
def load_dropdown_data():
    df = pd.read_parquet("processed_restaurant_data.parquet")  # Smaller than loading full CSV/JSON
    return df[['city', 'category']].dropna()

df = load_dropdown_data()

st.set_page_config(page_title="Yelp Rating Predictor", page_icon="🍽️")

st.title("🍽️ Predict Yelp Rating for Your Restaurant")

st.markdown("Give us your business details and we'll predict your expected average Yelp rating ⭐")

# Input form
with st.form("prediction_form"):
    city = st.selectbox("City", sorted(df['city'].unique()))
    category = st.selectbox("Category", sorted(df['category'].unique()))
    reviews_per_year = st.slider("Number of reviews per year", 0, 1000, 100)
    checkins_per_year = st.slider("Number of check-ins per year", 0, 2000, 300)
    sentiment = st.slider("Average sentiment score", -1.0, 1.0, 0.5)

    price_range = st.selectbox("Price Range ($ = 1 to $$$$ = 4)", [1, 2, 3, 4])
    has_delivery = st.selectbox("Delivery Available?", [True, False])
    has_takeout = st.selectbox("Takeout Available?", [True, False])
    has_creditcard = st.selectbox("Accepts Credit Cards?", [True, False])
    caters = st.selectbox("Caters?", [True, False])
    outdoor_seating = st.selectbox("Outdoor Seating?", [True, False])
    has_parking = st.selectbox("Parking Available?", [True, False])
    has_wifi = st.selectbox("WiFi", ["no", "free", "paid"])
    alcohol = st.selectbox("Alcohol", ["none", "beer_and_wine", "full_bar"])

    submit = st.form_submit_button("🚀 Predict Rating")

# Prediction
if submit:
    input_df = pd.DataFrame([{
        "city": city,
        "category": category,
        "reviews_per_year": reviews_per_year,
        "checkins_per_year": checkins_per_year,
        "sentiment": sentiment,
        "has_delivery": has_delivery,
        "has_takeout": has_takeout,
        "has_creditcard": has_creditcard,
        "caters": caters,
        "outdoor_seating": outdoor_seating,
        "price_range": price_range,
        "has_parking": has_parking,
        "has_wifi": has_wifi,
        "alcohol": alcohol
    }])

    # Predict
    pred = model.predict(input_df)[0]
    st.success(f"⭐ Predicted Rating: **{round(pred, 2)} stars**")

