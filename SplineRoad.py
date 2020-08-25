import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
import scipy.integrate as integrate


def plotSpline(xd, yd, xt, yt, xs, ys):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(xs, ys, label='spline')
    ax.plot(xt, yt, label='true')
    ax.plot(xd, yd, 'o', label='data')
    ax.legend(loc='lower left', ncol=2)
    plt.show()


def function(x: float): return x * x  # np.cos(0.5 * x)+np.sin(x)


def csd1(x: float): return cs.derivative(1)(x)


# Функция для длины кривой
def cslf(x: float): return np.sqrt(csd1(x) ** 2 + 1)


def cslenght(a, b): return integrate.quad(cslf, a, b)[0]


def findX(x0, l, epsilon):
    h = 0.001
    lcalc = 0.0
    start = x0
    while np.abs(l - lcalc) > epsilon:
        lcalc = cslenght(x0, start + h)
        start += h
    return start


# Проверка на длину. Просто так
def error(coord: np.ndarray, L):
    L_buf = 0
    for i in range(len(coord) - 1):
        L_buf += cslenght(coord[i], coord[i + 1])
    return np.abs(L_buf - L)


def equalSpaceCone(start, end, count_cone, epsilon):
    # Начало и конец
    x0 = start
    xN = end
    # Длина кривой
    L = cslenght(x0, xN)
    # Растояние между конусами
    l = L / count_cone
    # Координаты конусов
    normal_coord = np.array([x0])
    for i in range(count_cone + 1):
        x = normal_coord[-1]
        x = findX(x, l, epsilon)
        normal_coord = np.append(normal_coord, x)
    e = error(normal_coord, L)
    return normal_coord, e


# Построение нормали

def cone_locate():
    cone_coord = np.array()
    return cone_coord


def normal(x, xx0):
    return cs(xx0) - (x - xx0) / (csd1(xx0) + 0.0001)


def tangent(x, xx0): return cs(xx0) + (x - xx0) * csd1(xx0)


def transform(x, y, s, c):
    return x * c + y * s, -x * s + y * c


def translate(x, y, xn):
    pass


def normal_plot(normal_coord: np.ndarray, s: float):
    fig, ax = plt.subplots(figsize=(7, 7))
    lc = np.array([])
    rc = np.array([])
    for i in range(count_cone):
        # Точки нормали
        xcn0 = -s
        ycn0 = 0
        xcn1 = s
        ycn1 = 0
        # Серидина отрезка
        xcc = normal_coord[i]
        ycc = cs(xcc)
        # Для переноса
        cosb = xcc / np.sqrt(xcc ** 2 + ycc ** 2)
        sinb = ycc / np.sqrt(xcc ** 2 + ycc ** 2)
        # Для поворота
        cosa = np.cos(np.arctan(csd1(xcc)))
        sina = np.sin(np.arctan(csd1(xcc)))
        d = np.sqrt(xcc ** 2 + ycc ** 2)
        # Цепочка пребразований
        # 1 Поворот на  угол a. Наш базовый отрезок теперь перпендикулярен к касательной
        xcn0, ycn0 = transform(xcn0, ycn0, sina, cosa)
        xcn1, ycn1 = transform(xcn1, ycn1, sina, cosa)
        # 2 Параллельный перенос на расстояние d. Теперь центр базисного отрезка совпадает с серединой касательной
        xcn0 = xcn0 + d * cosb
        ycn0 = ycn0 + d * sinb
        xcn1 = xcn1 + d * cosb
        ycn1 = ycn1 + d * sinb
        lc = np.append(lc, np.array([[xcn0, ycn0]]))
        rc = np.append(rc, np.array([[xcn1, ycn1]]))
        ax.plot(xs, cs(xs), c='Black')
        ax.plot(xcn0, ycn0, 'o', c='Red')
        ax.plot(xcn1, ycn1, 'o', c='Red')
        ax.plot(xcc, ycc, 'o', c='Black')
    ax.grid()
    plt.show()
    return lc, rc


# Для построения нормали к поверхности используется метод Ньютона.
# Сперва для точке на OX, которые расположены на расстоянии d друг от друга производится поворот на 90
# Далее идет парраллеьный переност двух точек на расстояние d, которое находится на расстоянии s от точки нормали
# Получается СНУ, которое решается методом Ньютона, однако сходимость зависит от константы d, так  как в качестве
# Нначального приближения берется тчока нормали. Придуман спосбо решения проблемы зависимости
# от d. Решается сперва для малого значения d, далее некоторым шагом вплоть до нашегоо значения d
count_data = 10
count_point = 50
epsilon = 0.01
count_cone = 70
xd = np.linspace(1, 10, count_data)
yd = function(xd)  # function
cs = interpolate.CubicSpline(xd, yd)
xs = np.linspace(1, 10, count_point)
# TODO  проверка на общую длину
normal_coord, error_lenght = equalSpaceCone(xs[0], xs[-1], count_cone, epsilon)
print(normal_coord)
print("error=", error_lenght)

# data = pd.DataFrame({'X': xs, 'Y': cs(xs)})
# print(data)
# data.to_csv("data.csv")

left_cone, right_cone = normal_plot(normal_coord, s=1)
print(left_cone)
print(right_cone)
