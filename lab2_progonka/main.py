from math import exp, sqrt, floor
import numpy as np
import scipy.linalg as sl

from decimal import Decimal

TRACE_SIZE = 10
NODES = 81920


def k_a(x):
    return x


def q_a(x):
    return exp(-x)


def f_a(x):
    return x * x * x


def k_b(x):
    return x * x + 1


def q_b(x):
    return exp(-x)


def f_b(x):
    return 1


def find_model_problem_analytycal_trace(x_0, u_0, u_1):
    lambda_a = sqrt(q_a(x_0) / k_a(x_0))
    lambda_b = sqrt(q_b(x_0) / k_b(x_0))
    mu_a = f_a(x_0) / q_a(x_0)
    mu_b = f_b(x_0) / q_b(x_0)
    a_11 = exp(-lambda_a * x_0) - exp(lambda_a * x_0)
    a_12 = exp(lambda_b * (2 - x_0)) - exp(lambda_b * x_0)
    a_21 = k_a(x_0) * lambda_a * (exp(lambda_a * x_0) + exp(-lambda_a * x_0))
    a_22 = k_b(x_0) * lambda_b * (exp(lambda_b * (2 - x_0)) + exp(lambda_b * x_0))
    b_1 = mu_b - mu_a + (mu_a - u_0) * exp(lambda_a * x_0) - (mu_b - u_1) * exp(lambda_b * (1 - x_0))
    b_2 = k_a(x_0) * lambda_a * (u_0 - mu_a) * exp(lambda_a * x_0) + k_b(x_0) * lambda_b * (u_1 - mu_b) * exp(
        lambda_b * (1 - x_0))
    c_1 = (((u_0 - mu_a) * a_11 - b_1) * a_22 - ((u_0 - mu_a) * a_21 - b_2) * a_12) / (a_11 * a_22 - a_12 * a_21)
    c_2 = (b_1 * a_22 - b_2 * a_12) / (a_11 * a_22 - a_12 * a_21)
    c_3 = (b_2 * a_11 - b_1 * a_21) / (a_11 * a_22 - a_12 * a_21)
    c_4 = (u_1 - mu_b) * exp(lambda_b) - c_3 * exp(2 * lambda_b)

    ans = list()
    for i in range(TRACE_SIZE + 1):
        x = i / TRACE_SIZE
        if x < x_0:
            ans.append([x, c_1 * exp(lambda_a * x) + c_2 * exp(-lambda_a * x) + mu_a])
        else:
            ans.append([x, c_3 * exp(lambda_b * x) + c_4 * exp(-lambda_b * x) + mu_b])

    return ans


def find_numerical_solution(nodes, x_0, u_0, u_1, is_model):
    h = 1 / nodes
    n_a = floor(x_0 * nodes)
    n_b = n_a + 1
    u = 1
    l = 2
    d = np.zeros(nodes)
    ab = np.zeros((u + l + 1, nodes))
    if is_model:
        d[0] = u_0
        d[nodes - 1] = u_1
        for i in range(1, n_a):
            d[i] = -f_a(x_0) * h * h
        for i in range(n_b + 1, nodes - 1):
            d[i] = -f_b(x_0) * h * h

        # Начальные условия
        ab[1][0] = 1
        ab[1][nodes - 1] = 1
        # u_n_a = u_n_b
        ab[1][n_a] = 1
        ab[0][n_a + 1] = -1
        # Равенство с k
        ab[0][n_b + 1] = k_b(x_0)
        ab[1][n_b] = -k_b(x_0)
        ab[2][n_b - 1] = -k_a(x_0)
        ab[3][n_b - 2] = k_a(x_0)
        for i in range(1, n_a):
            a_i = k_a(x_0)
            b_i = -(k_a(x_0) + k_a(x_0) + q_a(x_0) * h * h)
            c_i = k_a(x_0)
            ab[0][i + 1] = a_i
            ab[1][i] = b_i
            ab[2][i - 1] = c_i
        for i in range(n_b + 1, nodes - 1):
            a_i = k_b(x_0)
            b_i = -(k_b(x_0) + k_b(x_0) + q_b(x_0) * h * h)
            c_i = k_b(x_0)
            ab[0][i + 1] = a_i
            ab[1][i] = b_i
            ab[2][i - 1] = c_i
    else:
        # not model
        d[0] = u_0
        d[nodes - 1] = u_1
        for i in range(1, n_a):
            d[i] = -f_a(i * h) * h * h
        for i in range(n_b + 1, nodes - 1):
            d[i] = -f_b(i * h) * h * h
        d[n_a] = 0
        d[n_b] = 0

        # Начальные условия
        ab[1][0] = 1
        ab[1][nodes - 1] = 1
        # u_n_a = u_n_b
        ab[1][n_a] = 1
        ab[0][n_a + 1] = -1
        # Равенство с k
        ab[0][n_b + 1] = k_b(n_b * h)
        ab[1][n_b] = -k_b(n_b * h)
        ab[2][n_b - 1] = -k_a(n_a * h)
        ab[3][n_b - 2] = k_a(n_a * h)

        for i in range(1, n_a):
            a_i = k_a(i * h + h / 2)
            b_i = -(k_a(i * h + h / 2) + k_a(i * h - h / 2) + q_a(i * h) * h * h)
            c_i = k_a(i * h - h / 2)
            ab[0][i + 1] = a_i
            ab[1][i] = b_i
            ab[2][i - 1] = c_i
        for i in range(n_b + 1, nodes - 1):
            a_i = k_b(i * h + h / 2)
            b_i = -(k_b(i * h + h / 2) + k_b(i * h - h / 2) + q_b(i * h) * h * h)
            c_i = k_b(i * h - h / 2)
            ab[0][i + 1] = a_i
            ab[1][i] = b_i
            ab[2][i - 1] = c_i
        # print_matrix(ab)
        # print()
        # print_matrix(d)
        # print()
        # print(d)
        # print()
        # rint(h)
    '''
    print(n_a, h * n_a)
    print(n_b, h * n_b)
    a = np.zeros((nodes, nodes))
    for j in range(nodes):
        for i in range(nodes):
            index = u + i - j
            if 0 <= index < u + l + 1:
                a[i][j] = ab[index][j]
    print_matrix(a)
    print()
    '''
    ans = sl.solve_banded((l, u), ab, d, overwrite_ab=True, overwrite_b=True)

    numerical_trace = list()
    trace_step = int((nodes - 1) / TRACE_SIZE)
    for i in range(TRACE_SIZE + 1):
        numerical_trace.append(((i * trace_step) / (nodes - 1), ans[i * trace_step]))
    return numerical_trace


def main():
    print("Краевые условия задачи: u(0) = 0; u(1) = 1")
    print("Точка разрыва x_0 = 0,525")
    # NODES = int(input("Введите число узлов сетки: "))
    print("Найдём решение модельной задачи c постоянными коэффицентами:")
    print("Число узлов: {}".format(NODES))
    x_0 = 0.525
    u_0 = 0
    u_1 = 1
    model_analytical_solution = find_model_problem_analytycal_trace(x_0, u_0, u_1)
    model_numerical_solution = find_numerical_solution(NODES + 1, x_0, u_0, u_1, is_model=True)
    print_answer_model(model_analytical_solution, model_numerical_solution)
    print()
    print("Найдём решение нашего уравнения теплопроводности с переменными коэффицентами:")
    real_numerical_solution = find_numerical_solution(NODES + 1, x_0, u_0, u_1, is_model=False)
    print_answer_real(real_numerical_solution)
    print()
    # print_multiple(x_0, u_0, u_1)


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


def print_multiple(x_0, u_0, u_1):
    for i in range(2, 7):
        nodes = 10 ** i + 1
        real_numerical_solution = find_numerical_solution(nodes, x_0, u_0, u_1, is_model=False)
        print("Число узлов сетки {}:".format(nodes))
        print_answer_real(real_numerical_solution)
        print()


def print_answer_model(model_analytical_solution, model_numerical_solution):
    for i in range(len(model_analytical_solution)):
        print("x = {0:8.4f},".format(model_analytical_solution[i][0]),
              "analytical = {0:8.4f},".format(model_analytical_solution[i][1]),
              "numerical = {0:8.4f},".format(model_numerical_solution[i][1]),
              "delta = {0:12.10f}".format(abs(model_numerical_solution[i][1] - model_analytical_solution[i][1])))


def print_answer_real(real_numerical_solution):
    for elem in real_numerical_solution:
        print("x = {0:8.4f},".format(elem[0]),
              "numerical = {0:8.4f}".format(elem[1]))


if __name__ == '__main__':
    main()
