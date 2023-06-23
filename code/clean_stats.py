import pandas as pd

df = pd.read_csv('stats copy.csv', header=None)

# 6 [
# 10 ]
# 13 ' (0:0  2:0  0:1  1:0  2:0  2:1  1:0  0:1)'


def clean_scores(row):
    # ' (0:0  2:0  0:1  1:0  2:0  2:1  1:0  0:1)'
    row = row.replace(' ', '').replace(':', '').replace('(', '').replace(')', '')
    return row


df.iloc[:, 13] = df.iloc[:, 13].apply(clean_scores)
df['M1 P1 Score'] = df.iloc[:, 13].apply(lambda row: int(row[0:1]))
df['M1 P2 Score'] = df.iloc[:, 13].apply(lambda row: int(row[1:2]))
df['M2 P1 Score'] = df.iloc[:, 13].apply(lambda row: int(row[2:3]))
df['M2 P2 Score'] = df.iloc[:, 13].apply(lambda row: int(row[3:4]))
df['M3 P1 Score'] = df.iloc[:, 13].apply(lambda row: int(row[4:5]))
df['M3 P2 Score'] = df.iloc[:, 13].apply(lambda row: int(row[5:6]))
df['M4 P1 Score'] = df.iloc[:, 13].apply(lambda row: int(row[6:7]))
df['M4 P2 Score'] = df.iloc[:, 13].apply(lambda row: int(row[7:8]))

df.drop(columns=[13], inplace=True)

# df = df[['P1 Type,P1 Target Speed,P1 Fine Tune,P2 Type,P2 Target Speed,P2 Fine Tuned, Slowest times,,,,,# Goals,# Matches,Scores,Puck starting locations,,,,']]

df.rename(columns={
    0: 'P1 Type',
    1: 'P1 Target Speed',
    2: 'P1 Fine Tune',
    3: 'P2 Type',
    4: 'P2 Target Speed',
    5: 'P2 Fine Tuned',
    6: 'Slowest times 1',
    7: 'Slowest times 2',
    8: 'Slowest times 3',
    9: 'Slowest times 4',
    10: 'Slowest times 5',
    11: '# Goals',
    12: '# Matches',
    13: 'Scores',
    14: 'Puck starting locations 1',
    15: 'Puck starting locations 2',
    16: 'Puck starting locations 3',
    17: 'Puck starting locations 4',
    18: 'Opponent'
}, inplace=True)

assert len(df) % 4 == 0, 'number of matches not grouped in four'
num_graders = len(df) // 4

for i in range(num_graders):
    grade = []
    for j in range(4):
        r = i * 4 + j
        g = df.loc[r, '# Goals']
        m = df.loc[r, '# Matches']
        grade.append(min((g / m), 1) * 25)
    for j in range(4):
        r = i * 4 + j
        df.loc[r, 'Grade'] = sum(grade)

df.sort_values(by=['P1 Target Speed', 'P2 Target Speed'], inplace=True)
df.to_csv('stat_cleaned_sort_by_speed.csv', index=False)
df.sort_values(by=['P1 Type', 'P2 Type'], inplace=True)
df.to_csv('stat_cleaned_sort_by_cart.csv', index=False)
