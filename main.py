from xmlrpc.client import Boolean
import pandas as pd
import numpy as np

# ETL Process

def drop_malformed(_x) -> Boolean:
    try:
        float(_x)
    except:
        return True
    return False


games = pd.read_csv('data/games.csv') # Reading CSV
games.rename(columns={'id_game': 'gid'}, inplace=True) # To have the same naming across both tables
games['gid'] = games['gid'].astype(np.int32) # To be able to use gid as array index

user = pd.read_csv('data/user.csv') # Reading CSV
user = user.rename(columns={'UserID': 'uid', 'GameID': 'gid'}) # To have the same naming across both tables
user['rating'].mask(user['rating'].apply(drop_malformed), inplace=True) # Mark malformed rows
user.dropna(inplace=True) # Drop malformed rows
user['gid'] = user['gid'].astype(np.int32) # To be able to use gid as array index
user['uid'] = user['uid'].astype(np.int32) # To be able to use uid as array index
user['rating'] = pd.to_numeric(user['rating']) # To be able to perform math operations (initially it is object)

def games_of_user(uid: int):
    return pd.merge(games.loc[games['gid'].isin( user.loc[user['uid'] == uid]['gid'] )], \
                    user.loc[user['uid'] == uid], on='gid')

# EDA

n = len(games['gid'].index)+1
corr_matrix = np.zeros((n, n))
cnt = np.zeros(n)
for usr in user['uid']:
    u = pd.merge(games_of_user(usr).loc[:, ['gid', 'rating']], games[['gid', 'genre']])
    for i, (game1, r1, g1) in u.iterrows():
        for j, (game2, r2, g2) in u.iterrows():
            if game1 != game2:
                corr_matrix[int(game1)][int(game2)] += (1 - 1/(r1 + r2)**2 / 4)
                cnt[int(game1)] += 1

# Adjust self-correlation
for game in games['gid']:
    corr_matrix[int(game)] = corr_matrix[int(game)] / cnt[int(game)]
    corr_matrix[int(game)][int(game)] = 1

# Adjust genre correlation
for genre in np.unique(games['genre']):
    d = games[games.genre == genre].loc[:, ['gid']]
    for i, gi in d.iterrows():
        for j, gj in d.iterrows():
            corr_matrix[int(gi)][int(gj)] = (corr_matrix[int(gi)][int(gj)] + .01) * 4


# Prediction 

def predict_for(uid: int):
    v = games_of_user(uid)['gid']
    A = np.array([[]])
    for g in v:
        if A.size == 0:
            A = corr_matrix[:, g]
        else:
            A = np.column_stack((A, corr_matrix[:, g]))
    return corr_matrix @ A

def predict_n_for(uid: int, n):
    predicted = predict_for(uid)
    games = games_of_user(uid)['gid'].to_numpy()
    arr = []
    for i in range(predicted.shape[0]):
        for j in range(predicted.shape[1]):
            arr.append((predicted[i][j], i, j,))
    arr.sort(key=lambda x: x[0])
    arr.reverse()
    arr = np.array(list(map(lambda x: x[1], arr)))
    return arr[~np.isin(arr, games)][:n]


def predict_for_games(games_: list, n: int):
    A = np.array([[]])
    for g in games_:
        if A.size == 0:
            A = corr_matrix[:, g]
        else:
            A = np.column_stack((A, corr_matrix[:, g]))
    predicted = corr_matrix @ A
    arr = []
    for i in range(predicted.shape[0]):
        for j in range(predicted.shape[1]):
            arr.append((predicted[i][j], i, j,))
    arr.sort(key=lambda x: x[0])
    arr.reverse()
    arr = np.array(list(map(lambda x: x[1], arr)))
    return arr[~np.isin(arr, games_)][:n]




if __name__ == '__main__':
    usern = 20
    print('User played:')
    print(*list(games_of_user(usern)["title"].to_numpy()), sep=', ', end='\n\n')
    print(f'User might also want to play:')
    print(*list(games[games['gid'].isin(predict_n_for(usern, 10))]['title'].to_numpy()), sep=', ')

    x = [5, 1, 7, 73, 108]
    print('User played:')
    print(*list(games[games.gid.isin(x)]["title"].to_numpy()), sep=', ', end='\n\n')
    print(f'User might also want to play:')
    print(*list(games[games['gid'].isin(predict_for_games(x, 5))]['title'].to_numpy()), sep=', ')