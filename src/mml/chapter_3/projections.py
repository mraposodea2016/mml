import logging
from copy import deepcopy

import numpy as np

from src.mml.chapter_2 import gaussian_elimination

# Override the INFO logs from gaussian elimination
logging.basicConfig(level=logging.WARNING, force=True)

# Tolerance for float comparisons
TOL = 10 ** -12


def gen_basis_matrix(span):
    """
    Given a list of vectors which span a subspace,
    find the basis of the space by Gaussian elimination, ie
    find the list of linearly independent vectors from the span list.
    """
    gauss = gaussian_elimination.run(deepcopy(span))['matrix']
    dim = gaussian_elimination.calc_dimension(gauss)
    return np.array([r[:dim] for r in span])


def project_to_basis(B, x):
    """
    Returns the projection vector of x onto the subspace U, spanned by
    the basis B.
    Also returns the projection matrix P, so that P x = pi_u.
    """
    B_T = np.transpose(B)
    x_b = np.matmul(B_T, x)
    B_T_B_inv = np.linalg.pinv(np.matmul(B_T, B))
    lamb = np.matmul(B_T_B_inv, x_b)
    pi_u = np.matmul(B, lamb)

    P = np.matmul(B, np.matmul(B_T_B_inv, B_T))
    return {'proj': pi_u,
            'proj_matrix': P}


def check_projection(pi_u, P, x, B):
    """
    Check:
    - if the displacement vector pi_u - x if orthogonal
    to all the basis vectors of U,
    - if the projection matrix may be applied n times to x
    without changing the projection.
    """
    orthogonal = all([abs(np.matmul(pi_u - x, np.transpose(B)[i])) < TOL for i in range(B.shape[1])])
    proj_squared = np.allclose(np.matmul(np.matmul(P, P), x), np.matmul(P, x))
    print(orthogonal and proj_squared)


if __name__ == '__main__':
    x = np.array([-1, -9, -1, 4, 1])
    span = np.transpose(np.array([[0, -1, 2, 0, 2],
                                  [1, -3, 1, -1, 2],
                                  [-3, 4, 1, 2, 1],
                                  [-1, -3, 5, 0, 7]]))
    B = gen_basis_matrix(span)

    projection = project_to_basis(B, x)

    check_projection(pi_u=projection['proj'],
                     P=projection['proj_matrix'],
                     x=x,
                     B=B)
