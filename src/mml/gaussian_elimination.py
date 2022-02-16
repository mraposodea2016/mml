import numpy as np


def swap_rows(u, p):
    for r in range(u.shape[0]):
        u[r][:len(p)] = [u[r][c] for c in p]
    return u


def sum_rows(u_tr, base_col, sum_col, m):
    u = u_tr.copy()
    for r in range(u.shape[0]):
        u[r][base_col] += u[r][sum_col] * m
    return u


u1 = np.array([[1, 1, -3, 1],
               [2, -1, 0, -1],
               [-1, 1, -1, 1]], dtype='float')

rows = u1.shape[0]
cols = u1.shape[1]
u1_tr = u1.copy()
print(u1_tr)
for i in range(rows):
    print(f"------{i}------")
    pos = np.array(np.argsort(abs(u1_tr[i][:(cols - i)])))
    u1_tr = swap_rows(u1_tr, p=pos)
    print(u1_tr)
    ref_col = cols - i - 1
    for j in range(ref_col):
        if u1_tr[i][j] != 0:
            mult = - u1_tr[i][j] / u1_tr[i][ref_col]
            u1_tr = sum_rows(u1_tr, j, ref_col, mult)
        print(u1_tr)