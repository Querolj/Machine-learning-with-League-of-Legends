import ready_data as rd
import random
import math

def saveStats(heroes, y):
    i = 0
    with open("matches_stats.csv", 'w') as f:
        for heroes_id in heroes:
            for id in heroes_id:
                f.write(str(id)+",")
            f.write(str(y[0][i])+","+str(y[1][i])+","+str(y[2][i])+"\n")
            i += 1

# charge les stats de 90193 matches
def loadStats():
    X = [0 for i in range(90193)]
    Y = [[0 for i in range(90193)] for j in range(3)]
    i = 0
    with open("matches_stats.csv") as f:
        for line in f:
            line=line.split('\n')[0].split(',')
            X[i] = [int(id) for id in line[:10]]
            Y[0][i] = int(line[10])
            Y[1][i] = float(line[11])
            Y[2][i] = int(line[12])
            i += 1
    return X, Y

def saveCounter(counter_values, total_couple):
    with open("champs_counter.csv", 'w') as f:
        for id1, id2 in counter_values.keys():
            f.write(str(id1)+","+str(id2)+","+str(counter_values[(id1, id2)])+","+str(total_couple[(id1, id2)])+"\n")

def loadCounter():
    counter_values = dict()
    total_couple = dict()
    sorted_id = sorted([i for i in rd.CHAMPS_ID.keys() if i != "id"])
    for i in range(len(sorted_id)):
        for j in range(0, len(sorted_id)):
            counter_values[(sorted_id[i],sorted_id[j])] = 0
            total_couple[(sorted_id[i],sorted_id[j])] = 0

    with open("champs_counter.csv") as f:
        for line in f:
            line=line.split('\n')[0].split(',')
            counter_values[(line[0], line[1])] = float(line[2])
            total_couple[(line[0], line[1])] = int(line[3])
    return counter_values, total_couple
def saveData():
    with open("champs_played_tot.csv", 'w') as f:
        for id in rd.CHAMPS_PLAYED_TOT.keys():
            f.write(str(id)+","+str(rd.CHAMPS_PLAYED_TOT[id])+"\n")

    #CHAMPS_STATS_RATIO
    with open("champs_stats_ratio.csv", 'w') as f:
        for id in rd.CHAMPS_PLAYED_TOT.keys():
            f.write(str(id))
            for ratio in rd.CHAMPS_STATS_RATIO[id]:
                f.write(","+str(ratio))
            f.write("\n")
    #CHAMPS_ITEMS_RATIO
    with open("champs_items_ratio.csv", 'w') as f:
        for id in rd.CHAMPS_ITEMS_RATIO.keys():
            f.write(str(id))
            for ratio in rd.CHAMPS_ITEMS_RATIO[id]:
                f.write(","+str(ratio))
            f.write("\n")

def loadData(limit = -1):
    sorted_id = []
    c = 0
    with open("cleaned_matches.csv") as f:
        for line in f:
            if limit > 0 and c >= limit:
                return sorted_id
            elif c >= 0:
                line=line.split('\n')[0].split(',')
                sorted_id.append([int(id) for id in line])
            c += 1
    c = 0
    with open("champs_played_tot.csv") as f:
        for line in f:
            if limit > 0 and c >= limit:
                return sorted_id
            elif c >= 0:
                line=line.split('\n')[0].split(',')
                rd.CHAMPS_PLAYED_TOT[int(line[0])] = int(line[1])
            c += 1
    c = 0
    with open("champs_stats_ratio.csv") as f:
        for line in f:
            if limit > 0 and c >= limit:
                return sorted_id
            elif c >= 0:
                line=line.split('\n')[0].split(',')
                id = int(line[0])
                for st in range(1,len(line)):
                    rd.CHAMPS_STATS_RATIO[id][st - 1] = float(line[st])
            c += 1

    with open("champs_items_ratio.csv") as f:
        for line in f:
            if limit > 0 and c >= limit:
                return sorted_id
            elif c >= 0:
                line=line.split('\n')[0].split(',')
                id = int(line[0])
                for st in range(1,len(line)):
                    rd.CHAMPS_ITEMS_RATIO[id][st - 1] = float(line[st])
            c += 1

    return sorted_id
