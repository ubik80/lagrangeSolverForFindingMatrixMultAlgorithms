import numpy as np
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt

# finds Wa, Wb, Wc, so that Wc*(Wa*vec(A) *. Wb*vec(B)) = vec(A*B)
# => A*B = mat(Wc*(Wa*vec(A) *. Wb*vec(B)))
# A, B are n x n matrices (to be multiplied). 
# Wa, Wb are p x n^2 matrices. 
# Wc is a n^2 x p matrix. 
# p is the number of products to use in the searched for algorithm.
# *. is the element-wise product.
# a = vec(A) is the n^2 long vector created by unrolling A. 
# A = mat(a) is the the matrix created from the vector a = vec(A).
def lagrangeSolver(n, p, numIters):
    nn=n**2

    errHist=np.zeros(numIters,dtype=float)

    a=np.zeros(nn,dtype=float)
    b=np.zeros(nn,dtype=float)
    c=np.zeros(nn,dtype=float)
    A=np.zeros([n,n],dtype=float)
    B=np.zeros([n,n],dtype=float)
    C=np.zeros([n,n],dtype=float)
    aWaveStar=np.zeros(p,dtype=float)
    bWaveStar=np.zeros(p,dtype=float)
    cWaveStar=np.zeros(p,dtype=float)
    cWave=np.zeros(nn,dtype=float)
    alpha=np.zeros(p,dtype=float)
    beta=np.zeros(p,dtype=float)
    gamma=np.zeros(nn,dtype=float)
    deltaA=np.zeros([p,nn],dtype=float)
    deltaB=np.zeros([p,nn],dtype=float)
    deltaC=np.zeros([nn,p],dtype=float)

    Wa=np.random.rand(p,nn)*2.0-1.0 # random start values
    Wb=np.random.rand(p,nn)*2.0-1.0
    Wc=np.random.rand(nn,p)*2.0-1.0

    numItersInTol=0

    for i in range(numIters):
        a=np.random.rand(nn)*2.0-1.0
        b=np.random.rand(nn)*2.0-1.0
        a/=np.linalg.norm(a,2)
        b/=np.linalg.norm(b,2)
        A=np.reshape(a,[n,n])
        B=np.reshape(b,[n,n])
        C=A.dot(B)
        c=np.array(C.reshape(nn),dtype=float)

        aWaveStar=Wa.dot(a)
        bWaveStar=Wb.dot(b)
        cWaveStar=aWaveStar*bWaveStar
        cWave=Wc.dot(cWaveStar)

        cMinCWave=c-cWave
        err=np.linalg.norm(cMinCWave,2)
        errHist[i]=err

        if err>0.0001 and cWaveStar.dot(cWaveStar)>0.0001:
            gamma=(cMinCWave.dot(cMinCWave)*cMinCWave
            /(cWaveStar.dot(cWaveStar)*cMinCWave.T.dot(cMinCWave)
            +cMinCWave.T.dot(Wc.dot((aWaveStar*aWaveStar+bWaveStar*bWaveStar)
            *Wc.T.dot(cMinCWave)))))

            deltaC=np.outer(gamma,cWaveStar)

            alpha=bWaveStar*Wc.T.dot(gamma)
            beta=aWaveStar*Wc.T.dot(gamma)

            deltaA=np.outer(alpha,a.T)
            deltaB=np.outer(beta,b.T)

            Wa+=deltaA
            Wb+=deltaB
            Wc+=deltaC

        if err<0.01:
            numItersInTol=numItersInTol+1
            if numItersInTol>1000:
                break
            else:
                numItersInTol=0

    return Wa,Wb,Wc,errHist,i #lagrangeSolver

n=3
p=23
maxNumIters=10000000

print("matrix size (n): "+str(n))
print("#multipliers (p): "+str(p))
print("#max iterations: "+str(maxNumIters))

Wa,Wb,Wc,errHist,i=lagrangeSolver(n,p,maxNumIters)

print("#done interations: "+str(i+1))
A=np.random.randint(-9,9,[n,n])
B=np.random.randint(-9,9,[n,n])
print("example calculation:")
print("A:")
print(A)
print("B:")
print(B)
print("C=A*B exakt:")
print(A.dot(B))
a=np.array(A.reshape(n**2),dtype=float)
b=np.array(B.reshape(n**2),dtype=float)
c=Wc.dot(Wa.dot(a)*Wb.dot(b))
C=c.reshape((n,n))
print("C=Wc*(Wa*b o Wb*b):")
print(C)
print("error plot:")
plt.plot(errHist)
