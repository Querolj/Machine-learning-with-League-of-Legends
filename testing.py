import ready_data as rd
import save_and_load as sl
import plotting as pl
import random
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from itertools import chain
from sklearn import tree

'''
Défini l'algorithme de prédiction utilisé (k_nearest_neighbors) et test

TODO : retirer les thresholds (facile) DONE
sauvegarder les matches (facile, juste les arrays et id du match) DONE
préparer une série de test, leur donner une structure
counter ?? faire une array 2D id*2 (avec doublon) = + si counter id2, sinon - (moyen)
ajouter quelques valeur pour similarité, y'a des trucs très bien (facile)
'''
def printArray2D(l):
    for i in l:
        # print(i)
        print(displayResult(i))

def displayResult(ret):
    (similarity, champs_id)= ret
    st = str(similarity)+" : ["
    for id in champs_id:
        st += rd.CHAMPS_ID[str(id)]+","
    return st+"]"

#Retourne le ratio win/lose
def getRatioWinLose(t):
    (w,l) = t
    total_win = 0
    total_lose = 0
    for i in range(len(w)):
        (dist_w, id_w) = w[i]
        (dist_l, id_l) = l[i]
        total_win += dist_w
        total_lose += dist_l
    return total_lose/(total_win+total_lose)

#distance +1 per different hero
def simple_distance(m1, m2, heroes_relation):
    dist = 5
    for i in range(10):
        if m1[i] != m2[i]:
            if m1[i] > m2[i]:
                id1 = m2[i]
                id2 = m1[i]
            else:
                id1 = m1[i]
                id2 = m2[i]
            dist += heroes_relation[(id1, id2)]
            # dist += 1
            if m1[i] in m2[:5]:
                dist -= 0.5
        else:
            dist -= 1
    return dist

def k_nearest_neighbors(x, points, dist_function, k, heroes_relation, display = False):
    K_winner = []
    K_loser = []
    x_loser = [x[i+5] if i<5 else x[i-5] for i in range(len(x))]

    ret_winner = []
    ret_loser = []
    for p in range(len(points)):
        K_winner.append((dist_function(x,points[p], heroes_relation),points[p]))
        K_loser.append((dist_function(x_loser,points[p], heroes_relation),points[p]))

    K_winner.sort()
    K_loser.sort()
    ret_winner = [(a,b) for (a,b) in K_winner]
    ret_loser = [(a,b) for (a,b) in K_loser]

    if display:
        print("loser : "+str(x_loser))
        print("medianes : ")
        print("2% winner : "+displayResult(ret_winner[round(len(ret_winner)*0.02)]))
        print("2% loser : "+displayResult(ret_loser[round(len(ret_loser)*0.02)]))
        print("10% winner : "+displayResult(ret_winner[round(len(ret_winner)*0.10)]))
        print("10% loser : "+displayResult(ret_loser[round(len(ret_loser)*0.10)]))
        print("25% winner : "+displayResult(ret_winner[round(len(ret_winner)*0.25)]))
        print("25% loser : "+displayResult(ret_loser[round(len(ret_loser)*0.25)]))
        print("50% winner : "+displayResult(ret_winner[round(len(ret_winner)/2)]))
        print("50% loser : "+displayResult(ret_loser[round(len(ret_loser)/2)]))
        print("75% winner : "+displayResult(ret_winner[round(len(ret_winner)*0.75)]))
        print("75% loser : "+displayResult(ret_loser[round(len(ret_loser)*0.75)]))
        print("98% winner : "+displayResult(ret_winner[round(len(ret_winner)*0.98)]))
        print("98% loser : "+displayResult(ret_loser[round(len(ret_loser)*0.98)]))
        print("----------------------------------\n")
    return (ret_winner,ret_loser)#(ret_winner[:k],ret_loser[:k])

def checkCounterValues(heroes, counter_values):
    counter_dist = 0
    for i in range(5):
        counter_dist += counter_values[(str(heroes[i]), str(heroes[i+5]))]

    return counter_dist

def displayCounterValues(counter_values, total_couple):
    for id1, id2 in counter_values.keys():
        if total_couple[(id1, id2)] > 60:
            local_counter = counter_values[(id1, id2)] - counter_values[(id2, id1)]
            print(CHAMPS_ID[id1]+" "+CHAMPS_ID[id2]+" : "+str(local_counter)+" , nb de matchup : "+str(total_couple[(id1, id2)]))

def displayHeroesList(heroes):
    st = "["
    for id in heroes:
        st += rd.CHAMPS_ID[str(id)]+","
    st += "]"
    print(st)


def setNone(heroes):
    not_none_index = []
    for i in range(10):
        if heroes[i] != None:
            not_none_index.append(i)
    heroes_sim = []
    for i in not_none_index:
        heroes_sim.append([heroes[j] if j != i else None for j in range(10)])
    return heroes_sim

# def researchMatches(heroes, matches_id):
def avg(l):
    return sum(l)/len(l)

def errorEvaluation(y, ys):
    return sum([(y[i] - ys[i]) ** 2 for i in range(len(y))]) * (1/len(y))

def decisionTree(X, Y):
    print("---------------- Testing decision Tree on 100000 matches : ")
    error_value = -1
    clf = tree.DecisionTreeRegressor()
    for i in range(3):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y[i], test_size = 0.33)
        av = avg(Y[i])
        clf = clf.fit(X_train, Y_train)
        prediction = clf.predict(X_test)
        error_value = errorEvaluation(prediction, Y_test)
        print(str(error_value) + ", taux d'erreur pondéré : "+str((error_value ** 0.5))+", average "+str(av))
    return error_value, prediction

# ny = nombre d'éléments à prédire, type = type de l'algorithme
def naiveBayes(X, Y, ny, type='B', limit=1000):
    X = X[:(limit + 1)]
    error_value = -1
    if ny == 0:
        Y = Y[:(limit + 1)]
    else:
        for i in range(ny):
            Y[i] = Y[i][:(limit + 1)]
    if type == 'B':
        print("---------------- Testing Naive Bayes BernoulliNB on "+str(limit)+" matches : ")
        gnb = BernoulliNB()
    elif type == 'G':
        print("---------------- Testing Naive Bayes GaussianNB on "+str(limit)+" matches : ")
        gnb = GaussianNB()
    elif type == 'M':
        print("---------------- Testing Naive Bayes MultinomialNB on "+str(limit)+" matches : ") # GaussianNB, , MultinomialNB
        gnb = MultinomialNB()

    if ny == 0:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33)
        av = avg(Y)
        Y_train = np.asarray(Y_train,dtype=np.int64)
        gnb = gnb.fit(X_train, Y_train)
        prediction = gnb.predict(X_test)
        error_value = errorEvaluation(prediction, Y_test)
        print(str(error_value) + ", taux d'erreur pondéré : "+str((error_value ** 0.5))+", average "+str(av))
    else:
        for i in range(ny):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y[i], test_size = 0.33)
            av = avg(Y[i])
            Y_train = np.asarray(Y_train,dtype=np.int64)
            gnb = gnb.fit(X_train, Y_train)
            prediction = gnb.predict(X_test)
            error_value = errorEvaluation(prediction, Y_test)
            print(str(error_value) + ", taux d'erreur pondéré : "+str((error_value ** 0.5))+", average "+str(av))
    return error_value, prediction

def initBatNum(Y):
    bat_num_win = dict()
    bat_num_lose = dict()
    for bat_num in Y:
        if bat_num in bat_num_win:
            bat_num_win[bat_num] += 1
        else:
            bat_num_win[bat_num] = 0
        if bat_num * -1 in bat_num_lose:
            bat_num_lose[bat_num * -1] += 1
        else:
            bat_num_lose[bat_num * -1] = 0
    return bat_num_win, bat_num_lose

def errorRatio(w_or_l, prediction, bat_st):
    (bat_num_win, bat_num_lose) = bat_st
    ratios = []
    for p in prediction:
        # print(p)
        if round(p) not in bat_num_win or bat_num_win[round(p)] == 0:
            win_bat = 0.5
        else:
            win_bat = bat_num_win[round(p)]
        if round(p) not in bat_num_lose or bat_num_lose[round(p)] == 0:
            lose_bat = 0.5
        else:
            lose_bat = bat_num_lose[round(p)]
        ratio = (win_bat/(lose_bat + win_bat)) * 100
        # print(str(win_bat) + " "+str(lose_bat)+" "+str(round(p)))
        ratios.append(ratio)
    error_ratio = 0
    i = 0
    for ratio in ratios:
        if ratio >= 50 and w_or_l[i]:
            error_ratio += 1
        elif ratio < 50 and not w_or_l[i]:
            error_ratio += 1
        i += 1
    print("erreur ratios : "+str((error_ratio/len(ratios))))

def shuffleXY(X, Y):
    w_or_l = [True for i in range(len(X))]
    for i in range(len(X)):
        r = random.randint(0,1)
        if r==0:
            tmp = X[i][5:]
            X[i][:5] = X[i][5:]
            X[i][5:] = tmp
            Y[i] *= -1
            w_or_l[i] = False
    return X, Y, w_or_l

def copyXY(X, Y):
    X_or = X.copy()
    Y_or = Y.copy()
    for i in range(3):
        Y_or[i] = Y[i].copy()
    return X_or, Y_or
def comparaisonAlgo(X, Y, is_shuffled = False):
    result = []
    nb_type = ['B', 'G', 'M']
    if is_shuffled:
        X_or, Y_or = copyXY(X, Y)

    for nb in nb_type:
        if is_shuffled:
            for i in range(3):
                X, Y[i], w_or_l = shuffleXY(X, Y[i])
        error_value, prediction = naiveBayes(X, Y, 3, nb, 30000)
        result.append(error_value)
        if is_shuffled:
            X, Y = copyXY(X_or, Y_or)
        X, Y = sl.loadStats()


    error_value, prediction = decisionTree(X, Y)
    result.append(error_value)
    if is_shuffled:
        pl.plot_histo(result, "Erreur des algorithmes avec shuffle")
    else:
        pl.plot_histo(result, "Erreur des algorithmes sans shuffle")

def replaceXWithCounter(X, counter_values):
    for x in X:
        x = [counter_values[(str(x[i]), str(x[i+5]))] for i in range(5)]
def testing():
    print("Loading data...")
    matches_id = sl.loadData()
    heroes_relation = rd.initHeroRelation()
    counter_values, total_couple = sl.loadCounter()
    X, Y = sl.loadStats()
    print("Data Loaded.")

    # X = replaceXWithCounter(X, counter_values)
    # comparaisonAlgo(X, Y, True)
    # errorRatio(w_or_l, prediction, bat_st)
    # X, Y, w_or_l = shuffleX(X, Y[2])
    bat_num_win, bat_num_lose = initBatNum(Y[2])
    X, Y, w_or_l = shuffleXY(X, Y[2])
    error_value, prediction = naiveBayes(X, Y, 0, 'B', 30000)
    # error_value, prediction = decisionTree(X, Y)
    errorRatio(w_or_l, prediction, (bat_num_win, bat_num_lose))


    # Test k_nearest_neighbors :
    
    # i = 0
    # k = 15
    # accuracy = 0
    # X, Y, w_or_l = shuffleXY(X, Y[2])
    # for x in X:
    #     if i > 0:
    #         print(accuracy/i)
    #     (w,l) = k_nearest_neighbors(x, matches_id, simple_distance, k, heroes_relation)
    #     ratio = getRatioWinLose((w[:k],l[:k]))
    #     if w_or_l[i] and ratio >= 50:
    #         accuracy += 1
    #     elif not w_or_l[i] and ratio < 50:
    #         accuracy += 1
    #     i += 1
    #
    # accuracy = accuracy / len(X)
    # print("accuracy : "+str(accuracy))

    # print("ratio "+str(k)+" : "+str(ratio*100)+" % de chance de gagner la partie.")





    # x = []
    # y = []
    #
    # for k in range(5, 101, 2):
    #     x.append(ratio)
    #     y.append(k)
    # pl.buildGraph(x,y)


    # print("Winners neighbors : \n")
    # printArray2D(w)
    # print("\nLosers neighbors : \n")
    # printArray2D(l)
    # print("ratio : "+str(ratio*100)+" % de chance de gagner la partie.\n")
    # counter_dist = checkCounterValues(t, counter_values)
    # print("Valeur de counter de la team : "+str(counter_dist))

testing()
