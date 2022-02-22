import numpy as np


def basis_change(T, A, S):
    return np.matmul(np.linalg.inv(T), np.matmul(A, S))


def check_basis_change(B, B_prime, C, C_prime, phi_B_C):
    P1 = basis_change(T=B,
                      A=np.array([[1, 0],
                                  [0, 1]], dtype='float'),
                      S=B_prime)

    P2 = basis_change(T=C_prime,
                      A=np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]], dtype='float'),
                      S=C)

    phi_B_prime_C_prime = basis_change(T=np.linalg.inv(P2),
                                       A=phi_B_C,
                                       S=P1)

    x_B_prime = np.array([2, 3])
    x_B = np.matmul(P1, np.transpose(x_B_prime))
    phi_x_C = np.matmul(phi_B_C, np.transpose(x_B))
    return all(np.matmul(P2, phi_x_C) == np.matmul(phi_B_prime_C_prime, x_B_prime))


if __name__ == '__main__':
    A_tilde = basis_change(T=np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]], dtype='float'),
                           A=np.array([[1, 1, 0],
                                       [1, -1, 0],
                                       [1, 1, 1]], dtype='float'),
                           S=np.array([[1, 1, 1],
                                       [1, 2, 0],
                                       [1, 1, 0]], dtype='float'))

    print(check_basis_change(B=np.array([[2, -1],
                                         [1, -1]], dtype='float'),
                             B_prime=np.array([[2, 1],
                                               [-2, 1]], dtype='float'),
                             C_prime=np.array([[1, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 1]], dtype='float'),
                             C=np.array([[1, 0, 1],
                                         [2, -1, 0],
                                         [-1, 2, -1]], dtype='float'),
                             phi_B_C=np.array([[1, -1],
                                               [1, 1],
                                               [2, -1]], dtype='float')))
