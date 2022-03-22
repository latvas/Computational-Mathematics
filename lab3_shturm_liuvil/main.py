from math import pi
import numpy as np
from scipy.optimize import fsolve
import scipy.linalg
from scipy import sparse

NODES = 2000
ANSWERS = 4


def q(x):
    if 0 <= x <= 0.5:
        return 2
    if 0.5 <= x <= 1:
        return 1
    raise Exception("Value of q(x) should be from 0 to 1")


def find_model_analytical_solution(x_0):
    ans = list()
    for i in range(1, ANSWERS + 1):
        ans.append(pi * pi * i * i / q(x_0))
    return ans


def generate_matrix(lambda_0, h, nodes, is_model, x_0):
    ab = np.zeros((3, nodes))
    ab[1][0] = 1 * h
    ab[1][nodes - 1] = 1 * h
    if is_model:
        for i in range(1, nodes - 1):
            a_i = 1
            b_i = -(1 + 1) + lambda_0 * q(x_0) * h * h
            c_i = 1
            ab[0][i + 1] = a_i
            ab[1][i] = b_i
            ab[2][i - 1] = c_i
    else:
        for i in range(1, nodes - 1):
            a_i = 1
            b_i = -(1 + 1) + lambda_0 * q(i * h) * h * h
            c_i = 1
            ab[0][i + 1] = a_i
            ab[1][i] = b_i
            ab[2][i - 1] = c_i

    return ab


def find_determinant(lambda_0, h, nodes, is_model, x_0):
    ab = generate_matrix(lambda_0, h, nodes, is_model, x_0)
    nodes = ab.shape[1]
    det = np.zeros(nodes)
    det[0] = ab[1][0]
    det[1] = ab[1][0] * ab[1][1] - ab[0][1] * ab[2][0]
    for i in range(2, nodes):
        det[i] = ab[1][i] * det[i - 1] - ab[2][i - 1] * ab[0][i] * det[i - 2]
    return det[nodes - 1]


def find_numerical_solution(nodes, is_model, x_0=0):
    h = 1 / (nodes - 1)
    roots = list()
    root_count = 0
    lambda_0 = - 1
    step = 1
    det_prev = find_determinant(lambda_0, h, nodes, is_model, x_0)
    while root_count < ANSWERS:
        det = find_determinant(lambda_0, h, nodes, is_model, x_0)
        if det * det_prev <= 0:
            root_count += 1
            # print("MAY BE ROOT!!")
            root = fsolve(func=find_determinant, x0=lambda_0 - step, args=(h, nodes, is_model, x_0))
            # print(root)
            roots.append(root[0])
        #print('{:6.1f},'.format(lambda_0), "{:15.7e}".format(det))
        det_prev = det
        lambda_0 += step
    return roots


def main():
    print("Задача Штурма-Лиувиля, Вариант 5")
    print("Число узлов:", NODES)
    print()
    print("Решим модельную задачу:")
    x_0 = 1
    model_analytical_solution = find_model_analytical_solution(x_0)
    model_numerical_solution = find_numerical_solution(NODES + 1, True, x_0)
    errors = list()
    for i in range(len(model_analytical_solution)):
        error = abs(model_numerical_solution[i] - model_analytical_solution[i])
        errors.append(error)
        print("analytical = {:10.5f}".format(model_analytical_solution[i]),
              "numerical = {:10.5f}".format(model_numerical_solution[i]),
              "delta = {:10.5e}".format(error))
    print()
    print("Норма погрешности: {:10.5e}".format(max(errors)))
    print()
    print("Решим нашу задачу Штурма-Лиувиля:")
    real_numerical_solution = find_numerical_solution(NODES+1, False)
    for i in range(len(real_numerical_solution)):
        print("numerical = {:10.5f}".format(real_numerical_solution[i]))


def print_matrix(a):
    format_string = '{:6.3f}'
    if len(a.shape) == 2:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a[i][j] != 0:
                    print(format_string.format(a[i][j]), end=" ")
                else:
                    print('{0:6d}'.format(int(a[i][j])), end=" ")
            print()
    elif len(a.shape) == 1:
        for i in range(a.shape[0]):
            if a[i] != 0:
                print(format_string.format(a[i]), end=" ")
            else:
                print('{0:6d}'.format(int(a[i])), end=" ")
        print()


if __name__ == '__main__':
    main()

    '''
    u = 1
    l = 1
    a = np.zeros((nodes, nodes))
    for j in range(nodes):
        for i in range(nodes):
            index = u + i - j
            if 0 <= index < u + l + 1:
                a[i][j] = ab[index][j]
    print_matrix(a)
    print()
    print(scipy.linalg.det(a))
    '''
