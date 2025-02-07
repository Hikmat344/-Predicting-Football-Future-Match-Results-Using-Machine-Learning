
import streamlit as st
import pandas as pd
import numpy as np
import joblib




# load the trained model, encoder and cleaned datasets
model_path = "models/fifa_match_predictor_pipeline.pkl"
encoder_path = "models/team_encoder.pkl"

try:
    qualified_df = pd.read_csv("cleaned_data/qualified_cleaned.csv")
    teams_df = pd.read_csv("cleaned_data/teams_cleaned.csv")
    matches_df = pd.read_csv("cleaned_data/matches_cleaned.csv")

    pipeline = joblib.load(model_path)
    team_encoder = joblib.load(encoder_path)
except Exception as e:
    st.error(f"Error loading files: {e}")



#to calculate_win_percentage
def calculate_win_percentage(df, team_col):
    win_counts = df.groupby(team_col)['match_result'].apply(lambda x: (x == 1).sum())
    total_matches = df[team_col].value_counts()
    win_percentage = win_counts / total_matches
    return win_percentage.fillna(0)

win_percentage_team1 = calculate_win_percentage(matches_df, 'team1')
win_percentage_team2 = calculate_win_percentage(matches_df, 'team2')


# now for the simulation we create a function


def simulate_tournament(model, team_encoder, qualified_df, num_simulations):
    teams = list(qualified_df['name'])
    results = {team: 0 for team in teams}
    
    # default features
    sim_year = 2018
    sim_cup = "FIFA World Cup"
    default_goal_diff = 0
    
    for _ in range(num_simulations):
        winners = teams.copy()
        while len(winners) > 1:
            next_round = []
            for i in range(0, len(winners), 2):
                if i + 1 >= len(winners):
                    next_round.append(winners[i])
                    continue
                team1_name = winners[i]
                team2_name = winners[i+1]
                team1_enc_val = team_encoder.transform([team1_name])[0] if team1_name in team_encoder.classes_ else 0
                team2_enc_val = team_encoder.transform([team2_name])[0] if team2_name in team_encoder.classes_ else 0
                team1_win_perc = win_percentage_team1.get(team1_name, 0)
                team2_win_perc = win_percentage_team2.get(team2_name, 0)
                # to retrieve confdr
                team1_conf = teams_df[teams_df['fifa_code'] == team1_name]['confederation'].iloc[0]
                team2_conf = teams_df[teams_df['fifa_code'] == team2_name]['confederation'].iloc[0]
                sim_features = pd.DataFrame([{
                    'team1_enc': team1_enc_val,
                    'team2_enc': team2_enc_val,
                    'year': sim_year,
                    'CupName': sim_cup,
                    'abs_goal_diff': default_goal_diff,
                    'team1_confederation': team1_conf,
                    'team2_confederation': team2_conf,
                    'team1_win_perc': team1_win_perc,
                    'team2_win_perc': team2_win_perc
                }])
                prediction = model.predict(sim_features)[0]
                if prediction == 1:
                    next_round.append(team1_name)
                elif prediction == -1:
                    next_round.append(team2_name)
                else:
                    next_round.append(np.random.choice([team1_name, team2_name]))
            winners = next_round
        results[winners[0]] += 1
    return results


# now make a ui for this


st.title("FIFA Match Predictor & Tournament Simulator")

st.header("Match Prediction")
team1_input = st.selectbox("Select Team 1", qualified_df['name'].unique())
team2_input = st.selectbox("Select Team 2", qualified_df['name'].unique())
# retreve confdr
team1_conf = teams_df[teams_df['fifa_code'] == team1_input]['confederation'].iloc[0]
team2_conf = teams_df[teams_df['fifa_code'] == team2_input]['confederation'].iloc[0]

if st.button("Predict Match Result"):
    try:
        team1_enc_val = team_encoder.transform([team1_input])[0]
        team2_enc_val = team_encoder.transform([team2_input])[0]
        team1_win_perc = win_percentage_team1.get(team1_input, 0)
        team2_win_perc = win_percentage_team2.get(team2_input, 0)
    except Exception as e:
        st.error(f"Error encoding team names: {e}")
        team1_enc_val, team2_enc_val = 0, 0
    input_features = pd.DataFrame([{
        'team1_enc': team1_enc_val,
        'team2_enc': team2_enc_val,
        'team1_confederation': team1_conf,
        'team2_confederation': team2_conf,
        'year': 2018,
        'CupName': "FIFA World Cup",
        'abs_goal_diff': 0,
        'team1_win_perc': team1_win_perc,
        'team2_win_perc': team2_win_perc

    }])
    
    match_result = pipeline.predict(input_features)[0]
    st.write(f"Predicted Result: {'Team 1 Wins' if match_result == 1 else ('Draw' if match_result == 0 else 'Team 2 Wins')}")
    
# for worldcup simulation
st.header("Simulate Tournament")
num_simulations = st.slider("Number of Simulations", min_value=1, max_value=1000, value=10)

if st.button("Run Tournament Simulation"):
    sim_results = simulate_tournament(pipeline, team_encoder, qualified_df, num_simulations)
    st.write("Tournament Simulation Results:")
    st.write("### Tournament Simulation Results (Wins out of 1000 simulations)")
    sim_results_df = pd.DataFrame(list(sim_results.items()), columns=["Team", "Wins"])

    sim_results_df = sim_results_df.sort_values(by="Wins", ascending=False)
    # to display the data
    st.write(sim_results_df)
    st.bar_chart(sim_results_df.set_index("Team"))