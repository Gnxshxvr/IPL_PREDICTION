# 🏏 IPL Victory Predictor

https://iplprediction-i25pns8kjsgxajv98xbmw3.streamlit.app/

A sleek Streamlit web app that predicts IPL match win probabilities using machine learning. This app forecasts the likelihood of the batting team winning based on current match conditions.

## ✨ Highlights

- **ML-Powered Predictions**: Uses a trained RandomForest model for accurate win probability forecasts.
- **Interactive UI**: Dark-themed interface with cricket stadium background and intuitive inputs.
- **Real-time Insights**: Computes run rates and provides match analysis.
- **IPL Teams & Venues**: Covers all major IPL teams and iconic stadiums.

## 🚀 What's inside

- `app.py` — Streamlit interface with prediction logic.
- `model.pkl` — Pre-trained RandomForest classifier.
- `team_encoder.pkl` & `venue_encoder.pkl` — **ESSENTIAL** label encoders for categorical data.
- `requirements.txt` — Python dependencies.

## 🧪 Features

- Select batting/bowling teams and venue from dropdowns.
- Input target score, current score, overs completed, and wickets down.
- Automatic calculation of run rate and required run rate.
- Win probability display with progress bars and insights.
- Input validation and edge case handling.

## 📦 Requirements

- Python 3.8+
- `streamlit`, `numpy`, `joblib`, `scikit-learn`, `pandas`

## ▶️ Run locally

1. Create a virtual environment:

```powershell
python -m venv venv
```

2. Activate it:

```powershell
venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Run the app:

```powershell
streamlit run app.py
```

5. Open the local link in your browser.

## 💡 Pro tips

- The model was trained on synthetic but realistic IPL data simulating real match scenarios.
- For best predictions, ensure overs completed is between 0.1 and 19.9.
- The app handles division by zero and invalid team selections gracefully.

## 📚 Notes

- Model accuracy depends on the quality of training data.
- This is a demo app; for production use, train on real IPL datasets.
- Share the app by deploying to Streamlit Cloud or any Python hosting service.

---

Built with passion for cricket and code! 🏏❤️

## 🔧 Technical Details

### Necessary Files (MUST include):
- `app.py` - Main application code
- `model.pkl` - Trained machine learning model
- `team_encoder.pkl` - Converts team names to numbers
- `venue_encoder.pkl` - Converts venue names to numbers
- `requirements.txt` - Python package dependencies

### Optional Files (can be removed):
- `README.md` - This documentation
- `.gitignore` - Git ignore rules
- `__pycache__/` - Python cache (auto-generated)

### Model Features:
The model uses 9 features to predict win probability:
1. `batting_team_encoded` - Batting team (numerical)
2. `bowling_team_encoded` - Bowling team (numerical)
3. `venue_encoded` - Match venue (numerical)
4. `target` - Target score to chase
5. `current_score` - Current batting score
6. `overs_completed` - Overs bowled (0-20)
7. `wickets_down` - Wickets fallen (0-10)
8. `run_rate` - Current run rate (calculated)
9. `required_run_rate` - Required run rate (calculated)

### Deployment Ready:
- ✅ No external API calls
- ✅ All dependencies in requirements.txt
- ✅ Model files included
- ✅ No sensitive data
- ✅ Works offline
