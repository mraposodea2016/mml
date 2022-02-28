import numpy as np
from src.mml.chapter_3 import projections
from itertools import product

TOL = 10 ** -12


def orthonormalize(B):
    """
    Starting from the first basis vector, project each subsequent basis
    vector onto the subspace spanned by the orthogonalized vectors preceding it,
    then normalize by dividing by its norm.
    """
    U = np.zeros(B.shape)
    U[0] = B[0] / np.linalg.norm(B[0])
    for i in range(1, len(B)):
        proj = projections.project_to_basis(np.transpose(U[:i]),
                                            B[i])['proj']
        orthogonal = B[i] - proj
        U[i] = orthogonal / np.linalg.norm(orthogonal)
    return U


def check_orthonormalization(U):
    """
    Check if all dot products between any two elements of the basis results in zero,
    and if all basis vectors have norm one.
    """
    orthogonal = all([abs(np.dot(u1, u2)) < TOL for u1, u2 in product(U, U) if not np.allclose(u1, u2)])
    normal = all([abs(np.linalg.norm(u) - 1) < TOL for u in U])
    print(orthogonal and normal)


if __name__ == "__main__":
    B = np.array([[1, 1, 1],
                  [-1, 2, 0]])
    U = orthonormalize(B)
    check_orthonormalization(U)
