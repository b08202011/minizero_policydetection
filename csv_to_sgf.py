import csv

def csv_to_sgf(csv_file, sgf_file):
    df = open(csv_file).read().splitlines()
    games = [i.split(',',1)[-1] for i in df]
    
    for game in games:
        moves_list = game.split(',')
        style = moves_list[0]
        moves_list=moves_list[1:]
        sgf = "(;GM[go_19x19]SZ[19]"
        if moves_list[-1][0]=='B':
           sgf += "PB["+style+"]PW[]"
        else:
           sgf += "PB[]PW["+style+"]"  
        for move in moves_list:
            player = move[0]
            x = move[2]
            y = move[3]
            if player == 'B':
                sgf += ";B[" + x + y+ "]"
            elif player == 'W':
                sgf += ";W[" + x + y+ "]"
        sgf += ")"    
        with open(sgf_file, 'a') as file:
            file.write(sgf+'\n')

# 使用示例
csv_file = str(input('csvfile'))
sgf_file = csv_file.replace('.csv','.sgf')
csv_to_sgf(csv_file, sgf_file)
