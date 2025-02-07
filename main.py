import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# first to do Data reading and Preprocessing

data_path = "data"  
matches_path = os.path.join(data_path, "matches.csv")
qualified_path = os.path.join(data_path, "qualified.csv")
teams_path = os.path.join(data_path, "teams.csv")

# loading datasets
matches_df = pd.read_csv(matches_path, parse_dates=['date'])
qualified_df = pd.read_csv(qualified_path)
teams_df = pd.read_csv(teams_path)

# to see the info about datasets
# st.sidebar.header("Dataset info")
# st.sidebar.write("Matches shape:", matches_df.shape)
# st.sidebar.write("Qualified Teams shape:", qualified_df.shape)
# st.sidebar.write("Teams shape:", teams_df.shape)

# to merge teams data with matches data for good features
# for team1
matches_df = matches_df.merge(teams_df[['name', 'confederation']], left_on='team1', right_on='name', how='left').drop(columns=['name'])
matches_df = matches_df.rename(columns={'confederation': 'team1_confederation'})

# for team2
matches_df = matches_df.merge(teams_df[['name', 'confederation']], left_on='team2', right_on='name', how='left').drop(columns=['name'])
matches_df = matches_df.rename(columns={'confederation': 'team2_confederation'})

# now to see the merged datset
# st.write("### Merged Matches Dataset ")
# st.write(matches_df.head())

# to fil the missing values in penalty , we assuming missing means no penalty occurd
matches_df['team1PenScore'] = matches_df['team1PenScore'].fillna(0)
matches_df['team2PenScore'] = matches_df['team2PenScore'].fillna(0)

# now to calculate the goal_diff and match_result
matches_df['goal_diff'] = matches_df['team1Score'] - matches_df['team2Score']
matches_df['match_result'] = matches_df['goal_diff'].apply(lambda x: 1 if x > 0 else (0 if x == 0 else -1))

# we extract only year from the date
matches_df['year'] = matches_df['date'].dt.year

# Now we will do Feature Engineering & Encodings

# Compute team win percentage
def calculate_win_percentage(df, team_col):
    win_counts = df.groupby(team_col)['match_result'].apply(lambda x: (x == 1).sum())
    total_matches = df[team_col].value_counts()
    win_percentage = win_counts / total_matches
    return win_percentage.fillna(0)

win_percentage_team1 = calculate_win_percentage(matches_df, 'team1')
win_percentage_team2 = calculate_win_percentage(matches_df, 'team2')

matches_df['team1_win_perc'] = matches_df['team1'].map(win_percentage_team1)
matches_df['team2_win_perc'] = matches_df['team2'].map(win_percentage_team2)

# let's make columns as strings
matches_df['team1'] = matches_df['team1'].astype(str)
matches_df['team2'] = matches_df['team2'].astype(str)
qualified_df['name'] = qualified_df['name'].astype(str)
teams_df['name'] = teams_df['name'].astype(str)

all_teams = pd.concat([matches_df['team1'], matches_df['team2'], qualified_df['name']]).unique()

# encode all teams
team_encoder = LabelEncoder()
team_encoder.fit(all_teams)

#  to make the team1 and team2 columns into numeric 
matches_df['team1_enc'] = matches_df['team1'].apply(lambda x: team_encoder.transform([x])[0])
matches_df['team2_enc'] = matches_df['team2'].apply(lambda x: team_encoder.transform([x])[0])


# to create a feature absolute goal difference
matches_df['abs_goal_diff'] = matches_df['goal_diff'].abs()


# to make the features and target seperate
feature_columns = ['team1_enc', 'team2_enc', 'year', 'CupName', 'abs_goal_diff', 'team1_confederation', 'team2_confederation', 'team1_win_perc', 'team2_win_perc']
X = matches_df[feature_columns]
y = matches_df['match_result']

# now we will creaste preprocessing pipeline for model


# define our numeric and categorical features
categorical_features = ['CupName', 'team1_confederation', 'team2_confederation']
numeric_features = ['team1_enc', 'team2_enc', 'year', 'abs_goal_diff', 'team1_win_perc', 'team2_win_perc']

# now we create a pipline for the numeric and sclae them ona standard 
numeric_transformer = Pipeline(steps=[ 
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for categorical features: impute and one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing', keep_empty_features=True)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# now we create a pipline for the categorical features and also do one hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# now create a complete pipline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=1, min_samples_split=10))
])


# now we train our model


# first we wil split our data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model 
pipeline.fit(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)
st.write(f"### Model Accuracy on Test Set: {test_accuracy:.2f}")

# we will save the trained pipeline and team encoder that we can use them n our predicions and simulations
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, os.path.join("models", "fifa_match_predictor_pipeline.pkl"))
joblib.dump(team_encoder, os.path.join("models", "team_encoder.pkl"))


# to create a folder for the cleaned data
os.makedirs("cleaned_data", exist_ok=True)

# save data
qualified_df.to_csv("cleaned_data/qualified_cleaned.csv", index=False)
teams_df.to_csv("cleaned_data/teams_cleaned.csv", index=False)
matches_df.to_csv("cleaned_data/matches_cleaned.csv", index=False)

print("model trained and data saved successfully!")



