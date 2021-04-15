import numpy as np
import matplotlib.pyplot as plt



def genTab(x_0,x_p, n):
    pom = (x_p - x_0) / (n - 1)
    macierz = np.array([x_0])

    for x in range(1, n, 1):
        macierz = np.block([
            [macierz, x * pom],
        ])
    return x, macierz

#print(genTab(0,1,4))


lp, tablica = genTab(0, 4, 100)
print (lp, tablica)
