import sklearn as sk
from sklearn.model_selection import KFold
from scipy.spatial import distance
from itertools import chain
from sklearn import tree
import numpy as np
import random
import math

'''
Sélectionne les données et apprend
'''


def getChampsId():
    content = dict()
    c = 0
    with open("champs.csv") as f:
        for line in f:
            line=line.split(',')
            content[line[1].split("\n")[0]] = line[0]
    return content
CHAMPS_ID = getChampsId()


CHAMPS_ITEMS_RATIO = dict()#keys = hero_id, value = ratio for each item (item,ratio)
CHAMPS_STATS = dict()#keys = hero_id, value = ratio for each item (item,ratio)
#contient les distances entre chaque couple de héro
HEROES_RELATION = dict()
#contient le nombre de fois chaque héro a été joué
CHAMPS_PLAYED_TOT = dict()#keys = hero_id, value = ratio for each item (item,ratio)
#contient les ratios des stats de chaque joueur
CHAMPS_STATS_RATIO = dict()
#contient les valeurs de counter de chaque champion
CHAMPS_COUNTER_VALUES = dict()
for i in CHAMPS_ID.keys():
    if i != 'id':
        id = int(i)
        CHAMPS_PLAYED_TOT[id] = 0
        CHAMPS_STATS_RATIO[id] = [0 for i in range(12)]
        CHAMPS_STATS[id] = [0 for j in range(35)]#35 types de stats
        CHAMPS_ITEMS_RATIO[id] = [0 for j in range(5000)]#valeur 5000 arbitraire
        CHAMPS_COUNTER_VALUES[id] = [0 for j in range(len(CHAMPS_ID.keys()) - 1)]

class Match:
    #party = 5 champs, party1 = winner
    def __init__(self, match_id, winner, loser):
        self.match_id = match_id
        #on aura besoin que de ca normalement
        self.heroes = [-1 for i in range(10)]
        self.party_win = [-1 for i in range(5)]
        if winner != None:
            self.addWinner(winner)
        self.party_lose = [-1 for i in range(5)]
        if loser != None:
            self.addLoser(loser)
        self.corrupted_data = False

    #0 = TOP, 1 = MID, 2 = JUNGLE, 3 = BOT_ADC, 4 = BOT_SUP
    def addWinner(self, participant):
        if participant[len(participant) - 1] == "TOP":
            self.party_win[0] = participant
            self.heroes[0] = participant[3]
        elif participant[len(participant) - 1] == "MID":
            self.party_win[1] = participant
            self.heroes[1] = participant[3]
        elif participant[len(participant) - 1] == "JUNGLE":
            self.party_win[2] = participant
            self.heroes[2] = participant[3]
        elif participant[len(participant) - 1] == "BOT":
            if participant[len(participant) - 2] == "DUO_CARRY":
                self.party_win[3] = participant
                self.heroes[3] = participant[3]
            elif participant[len(participant) - 2] == "DUO_SUPPORT":
                self.party_win[4] = participant
                self.heroes[4] = participant[3]
            elif participant[len(participant) - 2] == "DUO" or participant[len(participant) - 2] == "SOLO" \
            or participant[len(participant) - 2] == "NONE":
                if self.party_win[3] == -1:
                    self.party_win[3] = participant
                    self.heroes[3] = participant[3]
                elif self.party_win[4] == -1:
                    self.party_win[4] = participant
                    self.heroes[4] = participant[3]
                else:
                    self.corrupted_data = True
                    print("ERROR no place in team for DUO role")
            else:
                print("ERROR unknown position (winner BOT)"+str(participant))
                self.corrupted_data = True
        else:
            print("ERROR unknown position (winner)"+str(participant))
            self.corrupted_data = True

    def addLoser(self, participant):
        if participant[len(participant) - 1] == "TOP":
            self.party_lose[0] = participant
            self.heroes[5] = participant[3]
        elif participant[len(participant) - 1] == "MID":
            self.party_lose[1] = participant
            self.heroes[6] = participant[3]
        elif participant[len(participant) - 1] == "JUNGLE":
            self.party_lose[2] = participant
            self.heroes[7] = participant[3]
        elif participant[len(participant) - 1] == "BOT":
            if participant[len(participant) - 2] == "DUO_CARRY":
                self.party_lose[3] = participant
                self.heroes[8] = participant[3]
            elif participant[len(participant) - 2] == "DUO_SUPPORT":
                self.party_lose[4] = participant
                self.heroes[9] = participant[3]
            elif participant[len(participant) - 2] == "DUO" or participant[len(participant) - 2] == "SOLO"\
             or participant[len(participant) - 2] == "NONE":
                if self.party_lose[3] == -1:
                    self.party_lose[3] = participant
                    self.heroes[8] = participant[3]
                elif self.party_lose[4] == -1:
                    self.party_lose[4] = participant
                    self.heroes[9] = participant[3]
                else:
                    self.corrupted_data = True
                    print("ERROR no place in team for DUO role")
            else:
                print("ERROR unknown position (loser BOT)"+str(participant))
                self.corrupted_data = True
        else:
            print("ERROR unknown position (loser)"+str(participant))
            self.corrupted_data = True
    #Vérifie la validité du match
    def is_correct(self):
        if len(party_win) < 5 or len(party_lose) < 5:
            return False
        return True

    def displayMatch(self):
        if self.corrupted_data:
            print("\n-----------Info for CORRUPTED match "+str(self.match_id)+"-----------             /!\\/!\\/!\\")
        else:
            print("\n-----------Info for match "+str(self.match_id)+"-----------")
        print("Winners : \n")
        for p in self.party_win:
            if p == -1:
                print("no player for this position")
                continue
            print(str(p)+" "+CHAMPS_ID[p[2]])
        print("\n")
        print("Losers : \n")
        for p in self.party_lose:
            if p == -1:
                print("no player for this position")
                continue
            print(str(p)+" "+CHAMPS_ID[p[2]])
        print("Heroes ID :\n"+str(self.heroes))
def displayMatches(matches):
    for i in matches:
        matches[i].displayMatch()

#Create dict of Matches with winner/looser participants + some data for each matches
def assemble_matches(matches_data, participants, stats):
    matches = dict()
    for p in participants.keys():
        CHAMPS_PLAYED_TOT[int(participants[p][3])] += 1
        addItemChamps(stats, p, participants[p][3])
        if int(stats[p][0]) == 1:
            if participants[p][1] not in matches:
                matches[participants[p][1]] = Match(participants[p][1], participants[p],None)
            else:
                matches[participants[p][1]].addWinner(participants[p])
        else:
            if participants[p][1] not in matches:
                matches[participants[p][1]] = Match(participants[p][1], None, participants[p])
            else:
                matches[participants[p][1]].addLoser(participants[p])
    return matches

#Items stuff
def addItemChamps(stats, p_id, id):
    for item in stats[p_id][1:8]:
        CHAMPS_ITEMS_RATIO[int(id)][int(item)] += 1
    for s in range(22,55):#stats[p_id][21:57]:
        CHAMPS_STATS[int(id)][s- 22] += int(stats[p_id][s])

#Remove match in matches with -1 inside Match.heroes or if Match.corrupted_data
def removeCorruptedMatches(matches):
    clean_matches = dict()
    for i in matches:
        if not matches[i].corrupted_data:
            corrupted = False
            for id in matches[i].heroes:
                if id == -1:
                    corrupted = True
                    break
            if not corrupted:
                clean_matches[i] = matches[i]

    return clean_matches

#id, game_id, platformid, queueid, seasonid,duration, creation, version
def getMatches(limit = 0):
    content = dict()
    c = 0
    with open("matches.csv") as f:
        for line in f:
            if limit > 0 and c >= limit:
                return content
            elif c > 0:
                line=line.split(',')
                line[len(line) - 1] = line[len(line) - 1].split('\n')[0].split('\"')[1]
                line[2] = line[2].split('\"')[1]
                content[line[0]] = line[1:]
            c += 1
    return content
#id, match_id, player, championid, ss1, ss2, role, position
def getParticipants(limit = 0):
    content = dict()
    c = 0
    with open("participants.csv") as f:
        for line in f:
            if limit > 0 and c >= limit:
                return content
            elif c > 0:
                line=line.split(',')
                for i in range(len(line)):
                    line[i] = line[i].split('\n')[0].split('\"')[1]
                content[line[0]] = line
            c += 1
    return content

#id (participant), win (0 or 1), etc...
def getStats():
    content = dict()
    c = 100
    print("Loading stats...")
    with open("stats1.csv") as f:
        for line in f:
            c+=1
            # if c == 200101:
            #     return content
            line=line.split(',')
            for i in range(len(line)):
                line[i] = line[i].split('\n')[0].split('\"')[1]
            content[line[0]] = line[1:]

    with open("stats2.csv") as f2:
        for line in f2:
            line=line.split(',')
            c+=1
            if c == 1419271: #Données corrupted à partir de là
                return content
            for i in range(len(line)):
                line[i] = line[i].split('\n')[0].split('\"')[1]
            content[line[0]] = line[1:]
    print("Stats loaded.")
    return content

def convertListStrToInt(l):
    return [int(i) for i in l]
#matches = list of Match
#output = matches sorted by hero ID (similar to sort words by alphabetic order, each letter = each id from Match.heroes)
def sortData(matches):
    return sorted([convertListStrToInt(matches[i].heroes) for i in matches])

def reguleItemChamps():
    for id in CHAMPS_ITEMS_RATIO.keys():
        s = sum(CHAMPS_ITEMS_RATIO[id])
        CHAMPS_ITEMS_RATIO[id] = [i/s if s != 0 else 0 for i in CHAMPS_ITEMS_RATIO[id]]
def printItemChamps():
    for id in CHAMPS_ITEMS_RATIO:
        print("---------------------------------------------------------------------------------\nHero "+str(CHAMPS_ID[id])+"\n")
        print(CHAMPS_ITEMS_RATIO[id])
#STATS : 0 = totaldmg, 1= magicdmg, 2=physdmg, 3=truedmg, 4=largestcrit, 5=totaldmgchamp, 6=magicdmgchamp, 7=physdmgchamp
#8=truedmgchamp, 9=total_heal, 10=totalunithealed, 11=dmgselfmit, 12=dmgtoobj, 13=dmgtoturrets, 14=visionscore, 15=timecc
#16=totdmgtaken, 17=magicdmgtaken, 18=physdmgtaken, 19=truedmgtaken, 20=goldearned, 21=goldspent
def reguleStatsChamps():
    global CHAMPS_STATS_RATIO

    for id in CHAMPS_STATS.keys():
        if CHAMPS_PLAYED_TOT[id] == 0:
            continue
        #dmg dealt ratio
        total_dmg = CHAMPS_STATS[id][1] + CHAMPS_STATS[id][2] + CHAMPS_STATS[id][3]
        if total_dmg != 0:
            CHAMPS_STATS_RATIO[id][0] = CHAMPS_STATS[id][1]/total_dmg #magicdmg
            CHAMPS_STATS_RATIO[id][1] = CHAMPS_STATS[id][2]/total_dmg #physdmg
            CHAMPS_STATS_RATIO[id][2] = CHAMPS_STATS[id][3]/total_dmg #truedmg
        else:
            CHAMPS_STATS_RATIO[id][0] = 0 #magicdmg
            CHAMPS_STATS_RATIO[id][1] = 0 #physdmg
            CHAMPS_STATS_RATIO[id][2] = 0 #truedmg

        #dmg taken ratio
        total_dmg_taken = CHAMPS_STATS[id][17] + CHAMPS_STATS[id][18] + CHAMPS_STATS[id][19]
        if total_dmg_taken != 0:
            CHAMPS_STATS_RATIO[id][3] = CHAMPS_STATS[id][17]/total_dmg_taken #magicdmgtaken
            CHAMPS_STATS_RATIO[id][4] = CHAMPS_STATS[id][18]/total_dmg_taken #physdmgtaken
            CHAMPS_STATS_RATIO[id][5] = CHAMPS_STATS[id][19]/total_dmg_taken #truedmgtaken
        else:
            CHAMPS_STATS_RATIO[id][3] = 0 #magicdmgtaken
            CHAMPS_STATS_RATIO[id][4] = 0 #physdmgtaken
            CHAMPS_STATS_RATIO[id][5] = 0 #truedmgtaken

        #ratio using number of champs used
        CHAMPS_STATS_RATIO[id][6] = CHAMPS_STATS[id][9]/CHAMPS_PLAYED_TOT[id] #total_heal
        CHAMPS_STATS_RATIO[id][7] = CHAMPS_STATS[id][10]/CHAMPS_PLAYED_TOT[id] #totalunithealed
        CHAMPS_STATS_RATIO[id][8] = CHAMPS_STATS[id][11]/CHAMPS_PLAYED_TOT[id] #dmgselfmit
        CHAMPS_STATS_RATIO[id][9] = CHAMPS_STATS[id][14]/CHAMPS_PLAYED_TOT[id] #visionscore
        CHAMPS_STATS_RATIO[id][10] = CHAMPS_STATS[id][15]/CHAMPS_PLAYED_TOT[id] #timecc
        CHAMPS_STATS_RATIO[id][11] = CHAMPS_STATS[id][20]/CHAMPS_PLAYED_TOT[id] #goldearned
        #Pondère le vecteur
        ecart = math.sqrt(np.var(CHAMPS_STATS_RATIO[id]))
        CHAMPS_STATS_RATIO[id] = [CHAMPS_STATS_RATIO[id][i]/ecart if i > 5 else CHAMPS_STATS_RATIO[id][i]\
         for i in range(len(CHAMPS_STATS_RATIO[id]))]
        # print(CHAMPS_STATS_RATIO[id])


#output = dict : key are id_hero tuple, value is the similarity between these heros calculated with item,
#trickets and SS they usually wear
#Can be saved afterward INS jean
#Relation déterminée avec les objets ayant plus de 1% d'apparition
def calculheroRelation(id1, id2):
    if len(CHAMPS_STATS_RATIO[id1]) == 0 or len(CHAMPS_STATS_RATIO[id2]) == 0:
        return 100
    item_dist = distance.euclidean(CHAMPS_ITEMS_RATIO[id1], CHAMPS_ITEMS_RATIO[id2])
    stats_dist = distance.euclidean(CHAMPS_STATS_RATIO[id1], CHAMPS_STATS_RATIO[id2])
    return item_dist + stats_dist

def initHeroRelation():
    heroes_relation = dict()
    sorted_id = sorted([int(i) for i in CHAMPS_ID.keys() if i != "id"])

    for i in range(len(sorted_id)):
        for j in range(i+1, len(sorted_id)):
            heroes_relation[(sorted_id[i],sorted_id[j])] = calculheroRelation(sorted_id[i], sorted_id[j])
    return heroes_relation

#différence entre deux participants opposés, pour la même position. - : id2 counter id1, + : counter l'autre
#STATS : 0 = totaldmg, 1= magicdmg, 2=physdmg, 3=truedmg, 4=largestcrit, 5=totaldmgchamp, 6=magicdmgchamp, 7=physdmgchamp
#8=truedmgchamp, 9=total_heal, 10=totalunithealed, 11=dmgselfmit, 12=dmgtoobj, 13=dmgtoturrets, 14=visionscore, 15=timecc
#16=totdmgtaken, 17=magicdmgtaken, 18=physdmgtaken, 19=truedmgtaken, 20=goldearned, 21=goldspent, 24= total minion killed
#25=neutralminionskilled, 26=ownjunglekills, 27=enemyjunglekills, 28=totcctimedealt, 29=champlvl
def computeDiff(diff1, diff2):
    if (diff1 + diff2) != 0:
        return (diff1 - diff2)/(diff1 + diff2)
    else:
        return 0
def participantsDiff(id1, id2, stats):
    diff = 0
    st1 = stats[id1]
    st2 = stats[id2]
    diff += (int(st1[9]))/(int(st1[10]) + 1) - (int(st2[9]))/(int(st2[10]) + 1)
    diff += computeDiff(int(st1[41]), int(st2[41])) #goldearned
    diff += computeDiff(int(st1[44]), int(st2[44])) #total minion killed
    diff += computeDiff(int(st1[49]), int(st2[49])) #level

    # print(int(st1[21]) - int(st2[21]))
    # ecart = math.sqrt(np.var(diff)) #On pondère car les valeurs sont très différents entre elles
    # if ecart == 0:
    #     return 0
    # diff = [i/ecart for i in diff]
    # print(diff)
    # return sum(diff) # On fait la somme, et on obtient une distance negative ou positive pour savoir de quel coté penche le counter
    return diff

def initHeroCountersValues(matches, stats):
    counter_values = dict()
    total_couple = dict()
    sorted_id = sorted([i for i in CHAMPS_ID.keys() if i != "id"])
    for i in range(len(sorted_id)):
        for j in range(0, len(sorted_id)):
            counter_values[(sorted_id[i],sorted_id[j])] = 0#calculheroCounterValues(sorted_id[i], sorted_id[j])
            total_couple[(sorted_id[i],sorted_id[j])] = 0#calculheroCounterValues(sorted_id[i], sorted_id[j])

    for m in matches:
        for i in range(5):
            id1 = matches[m].heroes[i]
            id2 = matches[m].heroes[i+5]
            p_id1 = matches[m].party_win[i][0]
            p_id2 = matches[m].party_lose[i][0]
            counter_values[(id1, id2)] += participantsDiff(p_id1, p_id2, stats)
            total_couple[(id1, id2)] += 1

    for i in range(len(sorted_id)):
        for j in range(0, len(sorted_id)):
            if j != i and total_couple[(sorted_id[i],sorted_id[j])] != 0:
                counter_values[(sorted_id[i],sorted_id[j])] = counter_values[(sorted_id[i],sorted_id[j])] /\
                 total_couple[(sorted_id[i],sorted_id[j])]
                # print(counter_values[(sorted_id[i],sorted_id[j])])
    return counter_values, total_couple

def splitTrainingTest(matches, seed):
    random.seed(seed);
    training_matches = dict()
    test_matches = dict()
    for i in matches:
        r = random.uniform(0, 1)
        if r < 0.5:
            training_matches[i] = matches[i]
        else:
            test_matches[i] = matches[i]
    return (training_matches, test_matches)

def setGoldAvg(matches, stats):
    X = [0 for i in range(len(matches))] #contient liste des héros
    Y = [[0 for i in range(len(matches))] for j in range(3)] #contient diff de gold pour les gagnants pour chaque x
    i = 0
    for m in matches:
        gold_diff = 0
        ratio_diff = 0
        building_diff = 0
        for j in range(5):
            idw = matches[m].party_win[j][0]
            idl = matches[m].party_lose[j][0]
            gold_diff += int(stats[idw][40]) - int(stats[idl][40])
            building_diff += (int(stats[idw][42]) + int(stats[idw][43])) - (int(stats[idl][42]) + int(stats[idl][43]))
            if int(stats[idw][9]) == 0 and int(stats[idl][9]) == 0:
                ratio_diff = 0
            elif int(stats[idw][9]) == 0:
                ratio_diff -= (int(stats[idl][8])/int(stats[idl][9]))
            elif int(stats[idl][9]) == 0:
                ratio_diff += (int(stats[idw][8])/int(stats[idw][9]))
            else:
                ratio_diff += (int(stats[idw][8])/int(stats[idw][9])) - (int(stats[idl][8])/int(stats[idl][9]))

        X[i] = matches[m].heroes
        Y[0][i] = gold_diff
        Y[1][i] = ratio_diff
        Y[2][i] = building_diff

        i +=1
    return X,Y

# By default, take the 10 firsts match in the database
def init_data(seed, limit = -1):
    limit = limit * 10 + 1
    stats = dict()
    if limit != -1:
        participants = getParticipants(limit)
        matches_data = getMatches(limit)
    else:
        participants = getParticipants()
        matches_data = getMatches()
    stats = getStats()

    matches = assemble_matches(matches_data, participants, stats)
    matches = removeCorruptedMatches(matches)

    (training_matches, test_matches) = splitTrainingTest(matches, seed)
    X, Y = setGoldAvg(matches, stats)
    saveStats(X, Y)

    # reguleItemChamps()
    # reguleStatsChamps()
    return (matches, stats)


#------------test
# init_data(1, 100000)
