import numpy as np
import matplotlib.pyplot as plt



def genTab(x_0, x_p, n):
    pom = (x_p - x_0) / (n - 1)
    macierzJeden = np.array([1, 0 * pom + x_0])

    for i in range(1, n, 1):
        macierzJeden = np.block([
            [macierzJeden],
            [i + 1, i * pom + x_0],
        ])

    macierzDwa = np.array([1, 1, 2])

    for i in range(1, n-1, 1):
        macierzDwa = np.block([
            [macierzDwa],
            [i + 1, i + 1, i + 2],
        ])
    return macierzJeden, macierzDwa

def alokacjaPamieci(n):
    A = np.zeros((n, n))
    b = np.zeros((n, 1))
    return A, b

wezly = np.array([[1, 0],
                  [2, 1],
                  [3, 0.5],
                  [4, 0.75]])

elementy = np.array([[1, 1, 3],
                     [2, 4, 2],
                     [3, 3, 4]])

twb_L = 'D'
twb_R = 'D'

wwb_L = 0
wwb_R = 1


[wezel, element] = genTab(1, 2, 5)
wezel = np.array([[1, 0],
                  [2, 1],
                  [3, 0.5],
                  [4, 0.75]])

element = np.array([[1, 1, 3],
                     [2, 4, 2],
                     [3, 3, 4]])

plt.plot(wezel[:, 1], np.zeros((np.shape(wezel)[0], 1)))
plt.plot(wezel[:, 1], np.zeros((np.shape(wezel)[0], 1)), 'ro')
for i in range(np.shape(wezel)[0]):
    plt.text(wezel[i, 1] - 0.02, -0.01, "x" + str(int(wezel[i, 0])))
    plt.text(wezel[i, 1] - 0.01, -0.02, str(int(wezel[i, 0])), color="green", fontsize=16)
for i in range(np.shape(element)[0]):
    pom1 = element[i, 1]
    pom2 = element[i, 2]
    plt.text(wezel[pom1 - 1, 1] + 0.01, +0.001, "1", color="red", fontsize=16)
    plt.text(wezel[pom2 - 1, 1] - 0.05, +0.001, "2", color="red", fontsize=16)
    plt.text((wezel[pom1 - 1, 1]+wezel[pom2 - 1,1])/2, +0.02, str(element[i,0]), color="blue", fontsize=16)
plt.grid()
plt.show()

macierz = genTab(0,4,5)
print(wezel)
print(element)
