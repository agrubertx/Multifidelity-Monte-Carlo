from itertools import combinations
import numpy as np


# Get list of k-combinations from n_range.
def nchoosek(n_range, k):
    return np.array(list(combinations(n_range, k)))


# Helper function for optiMlevelCorr
def optiMlevelCorrHelper(sigma, rho, w, p, tog):

    k = sigma.shape[0]

    #to avoid index-out-of-range errors
    rho = np.append(rho, 0)

    #sanity check (see paper)
    assert(all(rho[1:k]**2 - rho[2:]**2 > 0))

    coeff = sigma[0]*rho[:-1]/sigma

    r = np.zeros(k)
    r[0] = 1
    r[1:] = np.sqrt(w[0]*(rho[1:k]**2 - rho[2:]**2) / (w[1:]*(1 - rho[1]**2)))

    #Best choice if m(i)>1 for all i
    m = np.zeros(k)
    m[0] = p / (r @ w)
    m[1:] = r[1:]*m[0]

    if tog:
        #Correction if some m(i)<1
        while any(m < 1):  # note that i <= k-1 and i+1 < j <= k
            assert(p > np.sum(w))  # otherwise it doesn't make sense
            i = np.where(m < 1)[0][0]  # get first index where m(i)<1

            m[i] = 1;
            r[i+1] = 1;
            if k > 2 and i < k-1:
                j = i+2
                r[j:] = np.sqrt( (w[i+1]*(rho[j:k]**2 - rho[j+1:]**2))
                                    / (w[j:]*(rho[i+1]**2 - rho[i+2]**2)) )

            m[i+1] = (p - np.sum(w[:i+1])) / (r[i+1:] @ w[i+1:])
            if k > 2 and i < k-1:
                j = i+2
                m[j:] = m[i+1]*r[j:]

    else:
        #Stupid hack way -- blows budget
        idx = np.where(m < 1)[0]
        m[idx] = 1

    v = sigma[0]**2 / m[0] + np.sum( (1./m[:-1] - 1./m[1:]) *
            (coeff[1:]**2 * sigma[1:]**2 - 2*coeff[1:] * rho[1:k] * sigma[0] * sigma[1:]) )

    Mlevel = np.floor(m).astype(int)

    return Mlevel, coeff, v


# Function which computes sampling strategy
def optiMlevelCorr(X, w, p, tog):
    k = X.shape[1]
    sigma = np.sqrt(np.var(X.T, axis=1, ddof=1))  #to match Matlab
    rho = [np.corrcoef(X.T)[0] if X.shape[1]>1 else np.corrcoef(X.T)]
#     vMC = sigma[0]**2 * w[0] / p

    return optiMlevelCorrHelper(sigma, rho, w, p, tog)


# Helper function for optimalOrderCorr
def isFeasible(rho, w):
    rho = np.append(rho, 0)
    y = 1
    for i in range(1,rho.shape[0]-1):
        y = y * (w[i-1]/w[i] > (rho[i-1]**2-rho[i]**2) / (rho[i]**2-rho[i+1]**2) )
    return y


# Function which computes optimal model set
def optimalOrderCorr(X, w, p, tog):

    k = X.shape[1]
    sigma = np.sqrt(np.var(X.T, axis=1, ddof=1))  #to match Matlab
    rho = np.corrcoef(X.T)[0]
    vMC = sigma[0]**2 * w[0] / p

    bestSet = 1
    bestV = vMC
    for i in range(k):
        allSets = nchoosek(range(1, k), i)
        for j in range(allSets.shape[0]):
            curSet = np.insert(allSets[j], 0, 0).astype(int)

            # order set w.r.t. correlation coefficient
            didx = np.flip(np.argsort(rho[curSet]))
            curSet = curSet[didx]
            assert(curSet[0] == 0)

            # check if feasability condition is satisified
            if not isFeasible(rho[curSet], w[curSet]):
                print(f'{curSet} is not feasible')
                continue

            # compute variance
            v = optiMlevelCorr(X[:, curSet], w[curSet], p, tog)[-1]

            print(f'{curSet} has variance {v}')
            if bestV > v:
                bestSet = curSet
                bestV = v

    return bestSet, bestV, vMC
