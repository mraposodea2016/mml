import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')


def gen_swap_positions(u_tr, index):
    return np.array(np.argsort(abs(u_tr[index:, index])))


def stop(u_tr, j, rows):
    return all([all(u_tr[r, :] == 0) for r in range(j, rows)])


def swap_rows(u_tr, p):
    rows = u_tr.shape[0]
    start_row = rows - len(p)
    for c in range(u_tr.shape[1]):
        u_tr[start_row:, c] = [u_tr[start_row + r][c] for r in p]
    logging.info(u_tr)


def scale_row(u_tr, row, factor):
    for c in range(u_tr.shape[1]):
        u_tr[row][c] *= factor
    logging.info(u_tr)


def sum_rows(u_tr, base_row, fixed_row, factor):
    for c in range(u_tr.shape[1]):
        u_tr[base_row][c] += u_tr[fixed_row][c] * factor
    logging.info(u_tr)


def run(u):
    u_tr = np.transpose(u)
    rows = u_tr.shape[0]
    cols = u_tr.shape[1]
    for j in range(cols):
        logging.info(f"------{j}------")
        logging.info(u_tr)
        if stop(u_tr, j, rows):
            break
        pos = gen_swap_positions(u_tr, j)
        swap_rows(u_tr, p=pos)
        scale_row(u_tr, j, 1 / u_tr[j, j])
        fixed_row = j
        for i in range(fixed_row + 1, rows):
            if u_tr[i][j] != 0:
                factor = - u_tr[i][j] / u_tr[fixed_row][j]
                sum_rows(u_tr, i, fixed_row, factor)
    return u_tr


if __name__ == '__main__':
    u1 = np.array([[1, 1, -3, 1],
                   [2, -1, 0, -1],
                   [-1, 1, -1, 1]], dtype='float')

    u_gauss = run(u=u1)
