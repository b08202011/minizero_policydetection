import csv
import random

def csv_to_sgf(csv_file, sgf_train_file,sgf_val_file):
    df = open(csv_file).read().splitlines()
    games = [i.split(',',2)[-1] for i in df]
    for game in games:
        moves_list = game.split(',')
        sgf = "(;GM[go_19x19]SZ[19]"
        for move in moves_list:
            player = move[0]
            x = move[2]
            y = move[3]
            if player == 'B':
                sgf += ";B[" + x + y+ "]"
            elif player == 'W':
                sgf += ";W[" + x + y+ "]"
        sgf += ")"    
        if random.random()<0.8:
            with open(sgf_train_file, 'a') as file:
                file.write(sgf+'\n')
        else:
            with open(sgf_val_file, 'a') as file:
                file.write(sgf+'\n')


# 使用示例
csv_file = str(input('csvfile'))
sgf_train_file = csv_file.replace('.csv','_train.sgf')
sgf_val_file = csv_file.replace('csv','_val.sgf')
csv_to_sgf(csv_file, sgf_train_file,sgf_val_file)
