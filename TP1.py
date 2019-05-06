# TP1 : ABALO Kokou IFI_P22 #
#================================#
# to execute : python3 TP1.py
#================================#
# ABALO Kokou IFI_P22 #
#================================#
# to execute : python3 TP1.py
#================================#
import numpy as np
import math
from scipy.linalg import svd
from scipy.spatial import distance


# Creer les 3 points avec leurs coordonnees 3D :

def enterPoint(N=3):
    p = np.empty((N,3))
# Recuperer les coordonnees dans la matrice p
    for i in range(N):
        print("Point ",i+1)
        p[i][0] = input("x :")
        p[i][1] = input("y :")
        p[i][2] = input("z :")
# Calcul de la transposee de la matrice p
    return np.asmatrix(np.array(p)).transpose()

def createCentroid(M):
  return np.mean(M, axis=1)

def covMatrix(A,Ca, B, Cb):
    row, col = A.shape
    H = np.zeros((row, col))
    for j in range(col):
        print("====================================================")
        print((A[:,j] - Ca)*(B[:,j]- Cb).transpose())
        print("====================================================")
        H = H + ((A[:,j] - Ca)*(B[:,j]- Cb).transpose())
    return H

def error(T, R, A, B):
    row, col = A.shape
    err = 0
    center = np.zeros((row,1))
# calcul de l'erreur estimee
    for j in range(col):
        V = R*A[:, j] + T - B[:, j]
        err = err + math.pow(distance.euclidean(center, V), 2)
    return err

def USVt_SVD(H):
    return svd(H)

def main():
    print("Entrer les coordonnées 3D des 3 points de l'ensemble A : ")
    A = enterPoint()
    print('matrix', A)
    Ca = createCentroid(A)
    print('matrix Ca', Ca)
    print("\n")
    print("Entrer les coordonnées 3D des 3 points de l'ensemble B : ")
    B = enterPoint()
    print('matrix', B)
    Cb = createCentroid(B)
    print('matrix Cb', Cb)
    print("\n")
    print("cov Matrix")
    H = covMatrix(A, Ca, B,Cb)
    print(H)
    print("\n")
    U, S, V = USVt_SVD(H)
    print("U", U.shape)
    print(U)
    print("\n")
    print("S", S.shape)
    print(S)
    print("\n")
    print("V", V.shape)
    print(V)
    print("\n")
    #Rotation
    R  = V*U.transpose()
    print("Rotation", R)
    print("\n")
    #Translation
    T = -1*R*Ca + Cb
    print("Transaltion", T)
    print("\n")
    #erreur
    err = error(T, R, A, B)
    print("Erreur : ", err)
if __name__ == '__main__':
    main()
