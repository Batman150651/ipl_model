import streamlit as st
import numpy as np

import joblib

model = joblib.load('ipl_model.pkl')
clf_trf = joblib.load('columntransformer.pkl')


teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Kings XI Punjab',
    'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]

city_list = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah'
]


st.title("🏏 IPL Win Predictor")

batting_team = st.selectbox("Select Batting Team", sorted(teams))
bowling_team = st.selectbox("Select Bowling Team", sorted([team for team in teams if team != batting_team]))
city = st.selectbox("Match City", sorted(city_list))

target = st.number_input('Target', min_value=0, max_value=400)
score = st.number_input("Current Score", min_value=0,max_value=400)
wickets = st.slider("Wickets Fallen", 0, 10)
overs = st.number_input("Overs Completed", 0.0, 20.0, step=0.1)


runs_left = target - score
balls_left = int(120 - (overs * 6))
crr = score / overs if overs > 0 else 0
rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0


input_data = np.array([[batting_team, bowling_team, city, runs_left, balls_left, 10 - wickets, target, crr, rrr]])

if st.button("Predict Win Probability"):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    current_run_rate = score / overs if overs > 0 else 0
    required_run_rate = (runs_left * 6) / balls_left if balls_left > 0 else 0
    input_dict = {
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [current_run_rate],
        'rrr': [required_run_rate]
    }

    import pandas as pd
    input_df = pd.DataFrame(input_dict)

    # Transform using the loaded ColumnTransformer
    transformed_input = clf_trf.transform(input_df)

    # Make prediction
    win_prob = model.predict_proba(transformed_input)[0][1]
    loss_prob = 1 - win_prob

    st.markdown(f"### 🟢 Win Probability: `{win_prob * 100:.2f}%`")
    st.markdown(f"### 🔴 Loss Probability: `{loss_prob * 100:.2f}%`")