import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import pandas as pd


# TODO

def plotSpline(xd, yd, xt, yt, xs, ys):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(xs, ys, label='spline')
    ax.plot(xt, yt, label='true')
    ax.plot(xd, yd, 'o', label='data')
    ax.legend(loc='lower left', ncol=2)
    plt.show()


def function(x): return np.sin(0.4 * x)


def csd1(x): return cs.derivative(1)(x)


# Функция для длины кривой
def cslcf(x): return np.sqrt(csd1(x) ** 2 + 1)


def cslenght(a, b): return integrate.quad(cslcf, a, b)[0]


def findX(x0, l, epsilon):
    h = 0.001
    lcalc = 0
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


# Построение нормали

def cone_locate():
#TODO
    cone_coord = np.array()
    return cone_coord

# Для построения нормали к поверхности используется метод Ньютона.
# Сперва для точке на OX, которые расположены на расстоянии d друг от друга производится поворот на 90
# Далее идет парраллеьный переност двух точек на расстояние d, которое находится на расстоянии s от точки нормали
# Получается СНУ, которое решается методом Ньютона, однако сходимость зависит от константы d, так  как в качестве
# Нначального приближения берется тчока нормали. Придуман спосбо решения проблемы зависимости
# от d. Решается сперва для малого значения d, далее некоторым шагом вплоть до нашегоо значения d







count_data = 10
count_point = 50
xd = np.linspace(1, 10, count_data)
yd = function(xd)  # function
cs = interpolate.CubicSpline(xd, yd)
xs = np.linspace(1, 10, count_point)
ys = cs(xs)
# TODO Без цикла


# TODO  проверка на общую длину
# Число конусов
count_cone = 40
# Начало и конец
x0 = xs[0]
xN = xs[-1]
# Длина кривой
L = cslenght(x0, xN)
# Растояние между конусами
l = L / count_cone
# Координаты конусов
#normal_coord = np.array([x0])
#for i in range(count_cone + 1):
#    x = normal_coord[-1]
#    x = findX(x, l, 0.01)
#    normal_coord = np.append(normal_coord, x)
#print(normal_coord)
#print("error=", error(normal_coord, L))
# data = pd.DataFrame({'X': xs, 'Y': cs(xs)})
# print(data)
# data.to_csv("data.csv")
xx = 1
xxn0 = xx - 0.001
xxn1 = xx + 0.001
xxt0 = xx - 0.001
xxt1 = xx + 0.001
yycs = cs(xx)
yycsd1 = csd1(xx)


# TODO Разница между концами векторами мы их переводим в нормальный

def normal(x, xx0):
    return cs(xx0) - (x - xx0) / (csd1(xx0) + 0.0001)


def tangent(x, xx0): return cs(xx0) + (x - xx0) * csd1(xx0)


def transform(x,y,s,c):
    return x*c + y * s,-x*s + y*c


def translate(x,y, xn):
    pass

print(np.dot(
    [xxn1 - xxn0, normal(xxn1, xx) - normal(xxn0, xx)],
    [xxt1 - xxt0, tangent(xxt1, xx) - tangent(xxt0, xx)]
))
# fig, ax = plt.subplots(figsize=(7, 7))
yyn0 = normal(xxn0, xx)
yyn1 = normal(xxn1, xx)
yyt0 = tangent(xxt0, xx)
yyt1 = tangent(xxt1, xx)

# ax.plot([xxn0, xxn1], [yyn0, yyn1], c='Red')
# ax.plot([xxt0, xxt1], [yyt0, yyt1], c='Green')
# ax.plot(xx,cs(xx),'o',c='Cyan')
# plt.show()
xc0 = 1
yc0 = 1
xc1 = 4
yc1 = 8


xcc = (xc0 + xc1) / 2
ycc = (yc0 + yc1) / 2
# Точки нормали
xcn0 = -0.5
ycn0 = 0
xcn1 = 0.5
ycn1 = 0
# TODO Неправильно определены cos и sin. В случае вычисления нормалей в качестве угла буду использовать
# производную от нормали

cosa = (xc1 - xc0)/(np.sqrt((xc0-xc1)**2 + (yc0-yc1)**2))
sina = (yc1 - yc0)/(np.sqrt((xc0-xc1)**2 + (yc0-yc1)**2))
cosb = xcc / np.sqrt(xcc**2 + ycc**2)
sinb = ycc / np.sqrt(xcc**2 + ycc**2)
xcn0, ycn0 = transform(xcn0,ycn0,sina,cosa)
xcn1, ycn1 = transform(xcn1,ycn1,sina,cosa)
d = np.sqrt(xcc**2 + ycc**2)
xcn0 = xcn0 + d*cosb
ycn0 = ycn0 + d*sinb
xcn1 = xcn1 + d*cosb
ycn1 = ycn1 + d*sinb
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([xc0,xc1],[yc0,yc1], c='Cyan')
ax.plot(xcn0,ycn0,'o', c='Red')
ax.plot(xcn1,ycn1,'o', c='Red')
ax.plot(xcc,ycc, 'o', c='Black')
ax.grid()
plt.show()


def normal_plot(normal_coord: np.ndarray, s: float):
    fig, ax = fig, ax = plt.subplots(figsize=(8, 8))
    s = 0.05
    fig, ax = plt.subplots(figsize=(7, 7))
    for i in range(30):
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
        cosa = 1/np.sqrt(1+csd1(xcc)**2)
        sina = csd1(xcc)/np.sqrt(1+csd1(xcc)**2)
        # Цепочка пребразований
        # 1 Поворот на  угол a. Наш базовый отрезок теперь перпендикулярен к касательной
        xcn0, ycn0 = transform(xcn0, ycn0, sina, cosa)
        xcn1, ycn1 = transform(xcn1, ycn1, sina, cosa)
        # 2 Параллельный перенос на расстояние d. Теперь центр базисного отрезка совпадает с серединой касательной
        xcn0 = xcn0 + d * cosb
        ycn0 = ycn0 + d * sinb
        xcn1 = xcn1 + d * cosb
        ycn1 = ycn1 + d * sinb
        ax.plot([xc0, xc1], [yc0, yc1], c='Cyan')
        ax.plot(xcn0, ycn0, 'o', c='Red')
        ax.plot(xcn1, ycn1, 'o', c='Red')
        ax.plot(xcc, ycc, 'o', c='Black')
    ax.grid()
    plt.show()

    # xxn0 = normal_coord - d
    # xxn1 = normal_coord + d
    # xxt0 = normal_coord - d
    # xxt1 = normal_coord + d
    # yyn0 = normal(xxn0, normal_coord)
    # yyn1 = normal(xxn1, normal_coord)
    # yyt0 = tangent(xxt0, normal_coord)
    # yyt1 = tangent(xxt1, normal_coord)
    # xs = np.linspace(1, 10, count_point)
    # ys = cs(xs)
    # ax.plot([xxn0, xxn1], [yyn0, yyn1], c='Red')
    # ax.plot([xxt0, xxt1], [yyt0, yyt1], c='Green')
    # ax.plot(normal_coord, cs(normal_coord), 'o', c='Cyan')
    # ax.plot(xs, ys, c='Magenta')
    # plt.show()
normal_plot(normal_coord,0.01)