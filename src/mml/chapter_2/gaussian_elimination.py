import numpy as np
import logging

# Change to logging.WARNING to stop logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Rounding Tolerance
TOL = 10 ** -12


# Format matrix printing
def matrix_float(x):
    if abs(x) <= TOL:
        return ' --- '
    elif x == 1:
        return '  1  '
    elif x > TOL:
        return " {0:.2f}".format(x)
    elif x < TOL:
        return "{0:.2f}".format(x)


np.set_printoptions(formatter={'float': matrix_float})


def gen_swap_positions(u_tr, col):
    # Returns the index positions that would sort a given column
    # by its elements' absolute values
    return np.flip(np.array(np.argsort(abs(u_tr[col:, col]))))


def stop(u_tr, j, rows):
    # Returns True if all rows below j are made up of zeros,
    # in which case the elimination has reached its end
    return all([all([abs(u_tr[r, c]) < TOL for c in range(u_tr.shape[1])]) for r in range(j, rows)])


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


def run(u_tr):
    rows = u_tr.shape[0]
    cols = u_tr.shape[1]
    for j in range(cols):
        logging.info(f"------{j}------")
        logging.info(u_tr)
        if stop(u_tr, j, rows):
            break
        pos = gen_swap_positions(u_tr, j)
        # Swap rows as to leave zeros at the bottom of the j column
        swap_rows(u_tr, p=pos)
        # Scale the j row so that the pivot element equals 1
        scale_row(u_tr, j, 1 / u_tr[j, j])
        fixed_row = j
        # Sum the remaining rows with multiples of the fixed row
        # so that their j-column elements are zero
        for i in range(fixed_row + 1, rows):
            if abs(u_tr[i][j]) > TOL:
                factor = - u_tr[i][j] / u_tr[fixed_row][j]
                sum_rows(u_tr, i, fixed_row, factor)
    return u_tr


def calc_dimension(u_gauss):
    # Calculate the dimension of space as the number of
    # pivots in Gauss matrix, ie the number of vectors that form its basis
    diag = np.diag(u_gauss)
    return sum(list(filter(lambda e: e == 1, diag)))


if __name__ == '__main__':
    u1 = np.array([[1, 0, 1],
                   [1, -2, -1],
                   [2, 1, 3],
                   [1, 0, 1]], dtype='float')

    u2 = np.array([[3, -3, 0],
                   [1, 2, 3],
                   [7, -5, 2],
                   [3, -1, 2]], dtype='float')

    u1_gauss = run(u_tr=u1)
    u2_gauss = run(u_tr=u2)

    dim1 = calc_dimension(u1_gauss)
    dim2 = calc_dimension(u2_gauss)
