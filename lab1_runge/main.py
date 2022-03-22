# Лабораторная работа во вычматам, пишем метод Рунге-Кутта
from math import exp
import sys

# sys.stdin = open("input.txt", "r")
# sys.stdout = open("output.txt", "w")

def print_analytycal_solutinon(A, B):
    c_1 = (2 * A - 5 * B) / 20
    c_2 = (2 * A + 5 * B) / 20
    print("y_1 =", 5 * c_1, "exp(-201x) +", 5 * c_2, "exp(-x)")
    print("y_2 =", -2 * c_1, "exp(-201x) +", 2 * c_2, "exp(-x)")


# print("A =", 5*c_1 + 5*c_2), print("B =", -2*c_1 + 2*c_2)

def analytical_solution(x, c_1, c_2):  # returns tuple(y_1, y_2)
    y_1 = 5 * c_1 * exp(-201 * x) + 5 * c_2 * exp(-x)
    y_2 = -2 * c_1 * exp(-201 * x) + 2 * c_2 * exp(-x)
    return (y_1, y_2)


def find_analytycal_trace(begin_point, end_point, A, B):
    ans = list()
    c_1 = (2 * A - 5 * B) / 20
    c_2 = (2 * A + 5 * B) / 20
    h = (end_point - begin_point) / 10
    if h <= 0:
        return ans
    for i in range(11):
        x = begin_point + i * h
        y = analytical_solution(x, c_1, c_2)
        ans.append((x, y[0], y[1]))
    return ans  # list of (x, y_1, y_2)


def f(x, y):  # x-number y-tuple(y_1, y_2)
    y_1 = -101 * y[0] + 250 * y[1]
    y_2 = 40 * y[0] - 101 * y[1]
    return (y_1, y_2)


def step_Runge_Kutta(x_n, y_n, h):  # x-number, y-tuple(y_1, y_2), h - number
    f_1 = f(x_n, y_n)
    f_2 = f(x_n + h / 2, (y_n[0] + f_1[0] * h / 2, y_n[1] + f_1[1] * h / 2))
    f_3 = f(x_n + h / 2, (y_n[0] + f_2[0] * h / 2, y_n[1] + f_2[1] * h / 2))
    f_4 = f(x_n + h, (y_n[0] + f_3[0] * h, y_n[1] + f_3[1] * h))
    y_npl1 = (y_n[0] + h / 6 * (f_1[0] + 2 * f_2[0] + 2 * f_3[0] + f_4[0]),
              y_n[1] + h / 6 * (f_1[1] + 2 * f_2[1] + 2 * f_3[1] + f_4[1]))
    return y_npl1


def Runge_Kutta(nodes, begin_point, end_point, y_1begin, y_2begin):
    y = (y_1begin, y_2begin)
    x = begin_point
    h = (end_point - begin_point) / nodes
    trace = [(x, y_1begin, y_2begin)]
    check = int(nodes / 10)
    for i in range(1, nodes + 1):
        x = begin_point + i * h
        y = step_Runge_Kutta(x, y, h)
        if (i % check == 0):
            trace.append((x, y[0], y[1]))
    return trace


D = 1.0
print("Решаем задачу Коши для системы уравнений:")
print("dy_1/dx = -101y_1 + 250y_2")
print("dy_2/dx =   40y_1 - 101y_2")
print("Ограничения имеют вид:")
print("y_1(0) = A, y_2(0) = B, 0 < x < D = {0:.3f}".format(D))

NODES = int(input("Число узлов сетки = "))
A = float(input("A = "))
B = float(input("B = "))

print("")
print("Аналитическое решение имеет вид:")
print_analytycal_solutinon(A, B)
print("")

# Найдём след аналитического решения
analytycal_trace = find_analytycal_trace(0, D, A, B)  # list of (x, y_1, y_2)
# for point in analytycal_trace: print(point)
# Решаем на сетке h
trace_h = Runge_Kutta(NODES, 0, D, A, B)
# Решаем на сетке 2h
# trace_2h = Runge_Kutta(NODES // 2, 0, D, A, B)
'''
for i in range(10):
	print("")
	print(analytycal_trace[i])
	print(trace_h[i])
'''

format_string = '{0:17e}'

print("ЧИСЛО УЗЛОВ СЕТКИ:", NODES, "\n")
print("Выведем результаты для y_1:")
print("x            ", end=" ")
for i in range(11):
    print(format_string.format(analytycal_trace[i][0]), end=" ")
print("\nАналитич.    ", end=" ")
for i in range(11):
    print(format_string.format(analytycal_trace[i][1]), end=" ")
print("\nРунге-Кутта  ", end=" ")
for i in range(11):
    print(format_string.format(trace_h[i][1]), end=" ")
print("\nОтн. Погрешн.", end=" ")
for i in range(11):
    print(format_string.format(abs(trace_h[i][1] - analytycal_trace[i][1]) / trace_h[i][1]), end=" ")

print("\n")
print("Выведем результаты для y_2:")
print("x            ", end=" ")
for i in range(11):
    print(format_string.format(analytycal_trace[i][0]), end=" ")
print("\nАналитич.    ", end=" ")
for i in range(11):
    print(format_string.format(analytycal_trace[i][2]), end=" ")
print("\nРунге-Кутта  ", end=" ")
for i in range(11):
    print(format_string.format(trace_h[i][2]), end=" ")
print("\nОтн. Погрешн.", end=" ")
for i in range(11):
    print(format_string.format(abs(trace_h[i][2] - analytycal_trace[i][2]) / trace_h[i][2]), end=" ")

print("")
