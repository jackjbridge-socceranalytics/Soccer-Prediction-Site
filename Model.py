import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Load in training set - stats from last season
shoot_stats = pd.read_csv('Shooting Stats.csv')
shoot_stats

# Set up ID map, to match team names to numbers, so we can use in the model
Team_ID_Map = {1: 'Arsenal', 2: 'Aston Villa', 3: 'Brighton', 4:'Burnley', 5: 'Chelsea', 6: 'Crystal Palace', 7:'Everton', 8: 'Fulham', 9: 'Leeds United', 10: 'Leicester City',
              11: 'Liverpool', 12: 'Manchester City', 13: 'Manchester Utd', 14: 'Newcastle Utd', 15: 'Sheffield Utd', 16: 'Southampton', 17: 'Tottenham', 18: 'West Brom',
              19: 'West Ham', 20: 'Wolves', 21: 'Bournemouth', 22: 'Norwich City', 23: 'Watford', 24: 'Cardiff City', 25: 'Huddersfield', 26: 'Stoke City', 27: 'Swansea City',
              29: 'Blackburn Rovers FC ', 30: 'Queens Park Rangers FC ', 31: 'Sheffield Wednesday FC '}
# Add a new column for each row in the training set, that marks whether a team won, lost, or tied
for i, row in shoot_stats.iterrows():

    Result = shoot_stats.at[i, 'Result']
    Team = shoot_stats.at[i, 'Team']
    Opponent = shoot_stats.at[i, 'Opponent']
    if Result == 'W':
        shoot_stats.at[i, 'winningTeam'] = list(Team_ID_Map.keys())[list(Team_ID_Map.values()).index(Team)]
        shoot_stats.at[i, 'losingTeam'] = list(Team_ID_Map.keys())[list(Team_ID_Map.values()).index(Opponent)]
    elif Result == 'D':
        shoot_stats.at[i, 'winningTeam'] = 0
        shoot_stats.at[i, 'losingTeam'] = 0
    else:
        shoot_stats.at[i, 'winningTeam'] = list(Team_ID_Map.keys())[list(Team_ID_Map.values()).index(Opponent)]
        shoot_stats.at[i, 'losingTeam'] = list(Team_ID_Map.keys())[list(Team_ID_Map.values()).index(Team)]
shoot_stats.head()

# Update the team names to be the numbers instead the names so we can use in the model
for i, row in shoot_stats.iterrows():
    Team = shoot_stats.at[i, 'Team']
    Opponent = shoot_stats.at[i, 'Opponent']
    shoot_stats.at[i, 'Team'] = list(Team_ID_Map.keys())[list(Team_ID_Map.values()).index(Team)]
    shoot_stats.at[i, 'Opponent'] = list(Team_ID_Map.keys())[list(Team_ID_Map.values()).index(Opponent)]
shoot_stats.head()

shoot_stats['winningTeam'] = shoot_stats['winningTeam'].astype(int)

# train the model using the training set
train_X = np.asarray(average_goals[['Team','Opponent', 'GA','GF']])
train_y = np.asarray(average_goals.winningTeam)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion = 'entropy', random_state = 42)
rfc.fit(train_X, train_y)

rfc_pred_train = rfc.predict(train_X)
print('Training Set Evaluation F1-Score=>',f1_score(train_y,rfc_pred_train,average='micro'))

# Load in the statistics for the games currently played this season
test_set = pd.read_csv('Premier2021.csv')
Team_ID_Map = {1: 'Arsenal', 2: 'Aston Villa', 3: 'Brighton', 4:'Burnley', 5: 'Chelsea', 6: 'Crystal Palace', 7:'Everton', 8: 'Fulham', 9: 'Leeds', 10: 'Leicester',
              11: 'Liverpool', 12: 'Man City', 13: 'Man Utd', 14: 'Newcastle', 15: 'Sheffield United', 16: 'Southampton', 17: 'Tottenham', 18: 'West Brom',
              19: 'West Ham', 20: 'Wolves', 21: 'AFC Bournemouth ', 22: 'Norwich City', 23: 'Watford FC ', 24: 'Cardiff City', 25: 'Huddersfield Town AFC ', 26: 'Stoke City FC ', 27: 'Swansea City AFC ',
              29: 'Blackburn Rovers FC ', 30: 'Queens Park Rangers FC ', 31: 'Sheffield'}
test_set1 = test_set[['Home','Away', 'Result']]
test_set1["Team 1"] = 0
test_set1["Team 2"] = 0
test_set1["GGD"] = 0
test_set1["winningTeam"] = "None"

# Go through and add columns for win/loss/draw, and also calculate the GF and GA statistics
for i, row in test_set1.iterrows():
    result = test_set1.at[i, 'Result']
    team1Score = result.split("-", 1)[0]
    team2Score = result.split("-", 1)[1]

    Team1 = str(test_set1.at[i, 'Home'])
    Team2 = str(test_set1.at[i, 'Away'])

    test_set1.at[i, "Team"] = list(Team_ID_Map.keys())[list(Team_ID_Map.values()).index(Team1)]
    test_set1.at[i,"Opponent"] = list(Team_ID_Map.keys())[list(Team_ID_Map.values()).index(Team2)]

    test_set1.at[i, 'GGD'] = abs(int(team2Score) - int(team1Score))
    test_set1.at[i, 'GA'] = int(team2Score)
    test_set1.at[i, 'GF'] = int(team1Score)
    if team1Score == team2Score:
        test_set1.at[i, 'winningTeam'] = 0
    elif team1Score > team2Score:
        test_set1.at[i, 'winningTeam'] = list(Team_ID_Map.keys())[list(Team_ID_Map.values()).index(Team1)]
    else:
        test_set1.at[i, 'winningTeam'] = list(Team_ID_Map.keys())[list(Team_ID_Map.values()).index(Team2)]

# Change the values from doubles to ints
test_set1['Team'] = test_set1['Team'].astype(int)
test_set1['Opponent'] = test_set1['Opponent'].astype(int)
test_set1['GA'] = test_set1['GA'].astype(int)
test_set1['GF'] = test_set1['GF'].astype(int)
test_set1.head()

# Test out the model on the test set, is around 70% accurate
test_set1['winningTeam'] = test_set1['winningTeam'].astype(int)
test_X = np.asarray(test_set1[['Team','Opponent', 'GA', 'GF']])
test_y = np.asarray(test_set1.winningTeam)

rfc_pred_test = rfc.predict(test_X)
print('Testing Set Evaluation F1-Score=>',f1_score(test_y,rfc_pred_test,average='micro'))
