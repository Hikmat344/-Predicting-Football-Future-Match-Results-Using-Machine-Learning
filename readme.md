FIFA Match Predictor & Tournament Simulator

Overview

This project predicts the outcome of international soccer matches using machine learning. It also simulates the 2018 FIFA World Cup multiple times to estimate the likelihood of different teams winning the tournament.

Features

Machine Learning Model: Trained to predict match outcomes based on historical data.

Tournament Simulation: Simulates World Cup matches using the trained model.

Interactive UI: Built with Streamlit for an easy-to-use experience.

Dataset

The dataset includes:

matches.csv: Historical match results (1950-2017)

teams.csv: Information about international teams and their confederations

qualified.csv: List of teams that qualified for the 2018 World Cup

Model & Approach

Algorithm: Random Forest Classifier

Feature Engineering:

Team win percentages

Goal differences

Confederation details

Match results encoding (Win/Loss/Draw)

Preprocessing:

One-hot encoding categorical variables

Standard scaling for numerical features

Installation & Setup

1. Clone the Repository

git clone https://github.com/your-username/fifa-match-predictor.git
cd fifa-match-predictor

2. Create virtual enviornment

python -m venev myenv

3. Activate virtual envoirnment

For Ubuntu/Linux: source myenv/bin/activate
For Window: myenv\Scripts\activate

4. Install Dependencies

pip install -r requirements.txt

5. Run main.py file

python main.py

6. Run the Application

streamlit run app.py

7. click on the given url

Usage

Predict a match result by selecting two teams.

Simulate the FIFA World Cup by running multiple simulations.

View Results in tabular and graphical format.

Future Improvements

Incorporating player-level data for better accuracy.

Enhancing model with deep learning techniques.

Adding real-time team performance updates.

License

This project is licensed under the MIT License.

