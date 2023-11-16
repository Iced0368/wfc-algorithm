import numpy as np
import random
import time

U = [i for i in range(2000)]

def random_pyset(k):
    return set(random.sample(U, k=k))

def npset_from_pyset(s):
    return np.array([1 if i in s else 0 for i in range(len(U))])

def test_equal(pyset, npset):
    for i in range(len(npset)):
        isPos = npset[i] > 0
        isIn = i in pyset
        if (isPos and isIn) or (not isPos and not isIn):
            continue
        return False
    return True


py_setA = random_pyset(1000)
np_setA = npset_from_pyset(py_setA)

py_setB = random_pyset(1000)
np_setB = npset_from_pyset(py_setB)

REP = 100000

s = time.time()
for i in range(REP):
    py_union = py_setA | py_setB
print("py union=", time.time()-s)

s = time.time()
for i in range(REP):
    py_intersect = py_setA & py_setB
print("py intersect=", time.time()-s)

print()

s = time.time()
for i in range(REP):
    np_union = np_setA + np_setB
print("np union=", time.time()-s)

s = time.time()
for i in range(REP):
    np_intersect = np_setA * np_setB
print("np intersect=", time.time()-s)

print(test_equal(py_union, np_union))
print(test_equal(py_intersect, np_intersect))
