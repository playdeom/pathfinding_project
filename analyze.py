import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os

def list_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

datas=[]
directory_path = "data/"
files = list_all_files(directory_path)
for file in files:
    if '.csv' in file:
        datas.append(file)
print(datas)
algolist=['A*','Flood Fill','Rotation Aware(0.5)','Rotation Aware(1)','UF SPS','MHIP','PSO']
for d in datas:
    df=pd.read_csv(d)
    # print(df.head())
    x=np.arange(len(algolist))
    moves,turns,time=[],[],[]
    for algo in algolist:
        moves.append(df[algo][0])
        turns.append(df[algo][1])
        time.append(df[algo][2])
    shortest_moves_value = min(moves)
    shortest_moves = [i for i, move in enumerate(moves) if move == shortest_moves_value]
    shortest_turns_value = min(turns)
    shortest_turns = [i for i, turn in enumerate(turns) if turn == shortest_turns_value]
    shortest_time_value = min(time)
    shortest_time = [i for i, t in enumerate(time) if t == shortest_time_value]
    
    print(f'=={d}==')
    print(f'Shortest in moves: {shortest_moves_value}')
    print(f'Shortest in turns: {shortest_turns_value}')
    print(f'Shortest in time: {shortest_time_value}\n')
    print(f'Best in moves: {", ".join([algolist[i] for i in shortest_moves])}')
    print(f'Best in turns: {", ".join([algolist[i] for i in shortest_turns])}')
    print(f'Best in time: {", ".join([algolist[i] for i in shortest_time])}\n')
    
    width=0.25
    plt.bar(x, moves, width, label='moves')
    plt.bar(x - width, turns, width, label='turns')
    plt.bar(x + width, time, width, label='time')
    plt.title(f'{d}')
    plt.xticks(x, algolist)
    plt.xlabel('cartegory')
    plt.ylabel('value')
    plt.legend()

    plt.show()