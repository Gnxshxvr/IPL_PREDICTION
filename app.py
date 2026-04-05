import streamlit as st
import numpy as np
import joblib
import pandas as pd

# =====================================================================================
# IPL VICTORY PREDICTOR - STREAMLIT WEB APP
# =====================================================================================
# This app predicts the winning probability of the batting team in IPL matches
# using a pre-trained RandomForest machine learning model.
#
# REQUIRED FILES:
# - model.pkl: Trained RandomForest classifier
# - team_encoder.pkl: Label encoder for team names
# - venue_encoder.pkl: Label encoder for venue names
# - requirements.txt: Python dependencies
#
# OPTIONAL FILES:
# - README.md: Project documentation
# =====================================================================================

# Step 1: Load the pre-trained model and encoders
# =====================================================================================
# The model was trained on historical IPL match data with these features:
# - batting_team_encoded: Numerical code for batting team
# - bowling_team_encoded: Numerical code for bowling team
# - venue_encoded: Numerical code for match venue
# - target: Target score to chase
# - current_score: Current batting team score
# - overs_completed: Number of overs bowled (0-20)
# - wickets_down: Number of wickets fallen (0-10)
# - run_rate: Current score / overs completed
# - required_run_rate: (target - current_score) / (20 - overs_completed)
#
# Output: Probability that batting team will win (0.0 to 1.0)
# =====================================================================================
model = joblib.load('model.pkl')
team_encoder = joblib.load('team_encoder.pkl')
venue_encoder = joblib.load('venue_encoder.pkl')

# Step 2: Define IPL teams and venues for dropdowns
# =====================================================================================
# These lists MUST match the teams and venues used during model training
# The order doesn't matter, but all names must be identical
# =====================================================================================
teams = ['Chennai Super Kings', 'Deccan Chargers', 'Delhi Capitals', 'Delhi Daredevils',
         'Gujarat Lions', 'Kings XI Punjab', 'Kochi Tuskers Kerala', 'Kolkata Knight Riders',
         'Mumbai Indians', 'Pune Warriors', 'Punjab Kings', 'Rajasthan Royals',
         'Rising Pune Supergiant', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad']

venues = ['Barabati Stadium', 'Brabourne Stadium', 'Chepauk Stadium', 'DY Patil Stadium',
          'Dr DY Patil Sports Academy', 'Eden Gardens', 'Feroz Shah Kotla',
          'Himachal Pradesh Cricket Association Stadium', 'Holkar Cricket Stadium',
          'M Chinnaswamy Stadium', 'Maharashtra Cricket Association Stadium', 'Nehru Stadium',
          'Punjab Cricket Association IS Bindra Stadium', 'Punjab Cricket Association Stadium',
          'Rajiv Gandhi International Stadium', 'Sardar Patel Stadium', 'Sawai Mansingh Stadium',
          'Vidarbha Cricket Association Stadium', 'Wankhede Stadium']

# Step 3: Set up Streamlit page configuration
# =====================================================================================
# This configures the app's appearance and behavior
# - page_title: Browser tab title
# - page_icon: Emoji icon in browser tab
# - layout: 'centered' for mobile-friendly design
# - initial_sidebar_state: Keep sidebar collapsed for cleaner look
# =====================================================================================
st.set_page_config(
    page_title="IPL Victory Predictor",
    page_icon="🏏",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Step 4: Custom CSS for dark theme and cricket stadium background
# =====================================================================================
# This creates a cricket-themed dark UI with:
# - Stadium background image from Unsplash
# - Golden text colors for cricket theme
# - Semi-transparent elements for readability
# - Custom styling for buttons and inputs
# =====================================================================================
st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), url('https://images.unsplash.com/photo-1540747913346-19bd32b51b6c?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .stTitle {
        color: #FFD700;
        text-align: center;
        font-size: 3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }
    .stMarkdown {
        color: #FFFFFF;
    }
    .stSelectbox, .stNumberInput, .stSlider {
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
        border: 1px solid #FFD700;
    }
    .stButton button {
        background-color: #FFD700;
        color: #000000;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .prediction-box {
        background-color: rgba(0,0,0,0.7);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #FFD700;
        text-align: center;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Step 5: Main title and live demo tag
# =====================================================================================
# HTML styling for the main title with cricket theme colors
# "LIVE DEMO" indicates this is a working demonstration
# =====================================================================================
st.markdown('<h1 class="stTitle">🏏 IPL Victory Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #FFD700; font-size: 1.2rem;">🔴 LIVE DEMO</p>', unsafe_allow_html=True)

st.divider()

# Step 6: Input section
# =====================================================================================
# Create a 2-column layout for better organization
# Left column: Team selections
# Right column: Match details
# =====================================================================================
st.subheader("Enter Match Details")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("🏏 Batting Team", teams, help="Select the team currently batting")
    bowling_team = st.selectbox("🎯 Bowling Team", teams, help="Select the team currently bowling")

with col2:
    venue = st.selectbox("🏟️ Venue", venues, help="Select the match venue")
    target = st.number_input("🎯 Target Score", min_value=1, max_value=500, value=200, help="Target score to chase")

st.markdown("---")
match_stage = st.radio(
    "Match Status",
    ("Match not started", "Match in progress"),
    horizontal=True,
    help="Choose if the match is still before start or already in progress."
)

match_started = match_stage == "Match in progress"

if match_started:
    col3, col4, col5 = st.columns(3)

    with col3:
        current_score = st.number_input("📊 Current Score", min_value=0, max_value=500, value=0, help="Current score of batting team.")

    with col4:
        overs_completed = st.slider("⏱️ Overs Completed", min_value=0.0, max_value=20.0, value=0.0, step=0.1, help="Overs completed so far.")

    with col5:
        wickets_down = st.number_input("❌ Wickets Down (optional)", min_value=0, max_value=10, value=0, help="Number of wickets fallen.")

    st.info("Match in progress: please enter the current score, overs, and wickets.")
else:
    current_score = 0
    overs_completed = 0.0
    wickets_down = 0
    st.info("Match has not started: prediction uses initial match conditions.")

st.divider()

# Step 7: Prediction button
# =====================================================================================
# use_container_width=True makes the button span the full width
# This creates a prominent call-to-action
# =====================================================================================
predict_button = st.button("🔮 Predict Winning Probability", use_container_width=True)

# Step 8: Prediction logic
# =====================================================================================
# This section only executes when the user clicks the prediction button
# =====================================================================================
if predict_button:
    # Input validation
    # =====================================================================================
    # Check for common user errors:
    # 1. Same team batting and bowling
    # 2. Invalid overs only when match has started
    # =====================================================================================
    if batting_team == bowling_team:
        st.error("❌ Batting team and bowling team cannot be the same!")
    elif match_started and overs_completed >= 20:
        st.error("❌ Match cannot have 20 or more overs completed!")
    else:
        # Compute derived features
        # =====================================================================================
        # These are calculated in real-time based on user inputs:
        # - run_rate: Current scoring rate
        # - required_run_rate: Required scoring rate to win
        # =====================================================================================
        if match_started:
            if overs_completed == 0:
                st.warning("⚠️ Overs completed cannot be zero in an ongoing match. Using 0.1 for calculation.")
                effective_overs = 0.1
            else:
                effective_overs = overs_completed

            run_rate = current_score / effective_overs
            required_run_rate = (target - current_score) / (20 - effective_overs) if (20 - effective_overs) > 0 else 0
        else:
            # Pre-match prediction uses starting conditions
            run_rate = 0
            required_run_rate = target / 20

        # Encode categorical variables
        # =====================================================================================
        # Convert team and venue names to numerical codes that the model understands
        # This is CRITICAL - the model only accepts numbers, not text
        # =====================================================================================
        batting_team_encoded = team_encoder.transform([batting_team])[0]
        bowling_team_encoded = team_encoder.transform([bowling_team])[0]
        venue_encoded = venue_encoder.transform([venue])[0]

        # Create input dataframe
        # =====================================================================================
        # Format the data exactly as the model was trained
        # All features must be in the correct order and format
        # =====================================================================================
        input_data = pd.DataFrame({
            'batting_team_encoded': [batting_team_encoded],
            'bowling_team_encoded': [bowling_team_encoded],
            'venue_encoded': [venue_encoded],
            'target': [target],
            'score': [current_score],
            'overs': [overs_completed],
            'wickets': [wickets_down],
            'run_rate': [run_rate],
            'required_run_rate': [required_run_rate]
        })

        # Make prediction
        # =====================================================================================
        # model.predict_proba() returns probabilities for both classes [lose, win]
        # We take [0][1] to get the probability of batting team winning
        # =====================================================================================
        win_probability = model.predict_proba(input_data)[0][1]  # Probability of batting team winning

        # Display results
        # =====================================================================================
        # Create a styled container for the prediction results
        # Show both batting and bowling team probabilities
        # Use progress bars for visual representation
        # =====================================================================================
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.subheader("🎉 Prediction Results")

        col_result1, col_result2 = st.columns(2)

        with col_result1:
            st.metric(f"🏏 {batting_team} Win Probability", f"{win_probability * 100:.1f}%")
            st.progress(win_probability)

        with col_result2:
            st.metric(f"🎯 {bowling_team} Win Probability", f"{(1 - win_probability) * 100:.1f}%")
            st.progress(1 - win_probability)

        st.markdown('</div>', unsafe_allow_html=True)

        # Additional insights
        # =====================================================================================
        # Provide cricket-specific analysis based on run rates
        # This adds value beyond just the probability numbers
        # =====================================================================================
        if run_rate > required_run_rate:
            st.success("📈 Batting team is ahead in run rate!")
        elif run_rate < required_run_rate:
            st.warning("📉 Batting team needs to accelerate!")
        else:
            st.info("⚖️ Run rates are balanced!")

else:
    # Default message when no prediction has been made
    # =====================================================================================
    # Guide the user on what to do next
    # =====================================================================================
    st.info("👆 Fill in the match details above and click 'Predict Winning Probability' to get the forecast.")

# Step 9: Footer
# =====================================================================================
# Simple footer with credits and technology stack
# =====================================================================================
st.divider()
st.markdown("""
    <p style="text-align: center; color: #FFD700;">
    Built with ❤️ using Streamlit | Powered by Machine Learning
    </p>
    """, unsafe_allow_html=True)
