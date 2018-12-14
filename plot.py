import matplotlib.pyplot as plt
import numpy as np

def buildGraph(x,y):
    plt.plot(x, y)

    plt.xlabel('ratio')
    plt.ylabel('K')
    plt.title('Evolution du ratio en fonction du k choisi')
    plt.grid(True)
    plt.savefig("testing_results_kn.png")
    plt.show()
