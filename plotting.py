import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def plot_histo(x, title):
    plt.bar([1,2,3,4], x, width=0.8)
    plt.xlabel('Algorihmes')
    plt.ylabel('Distance')
    plt.title(title)
    plt.xticks([1,2,3,4], ['BernoulliNB','GaussianNB','MultinomialNB', 'Decision Tree'])
    plt.show()
