import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt


def toCartesean(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y


def plot_spline(x_i, y_i):
    x =np .r_[x_i, x_i[0]]
    y = np.r_[y_i, y_i[0]]
    # TODO Проверка на самопересечение!!!!!! И 3 графика внутренний основной внешний
    # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    tck, u = interpolate.splprep([x, y], s=0, per=True)

    # evaluate the spline fits for 1000 evenly spaced distance values
    xx, yy = interpolate.splev(np.linspace(0, 1, 1000), tck)
    return xx, yy
    # plot the result
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(x, y, 'or')
    # ax.plot(xi, yi, '-b')


d = 6  # Размео трассы
# 1 Полряные координаты
# Основаня трасса
r_1 = np.array(np.random.randint(10,20,59))
phi_1 = np.radians(np.array(range(10,360,6)))
# Внешняя трасса
r_2 = r_1 + d
phi_2 = phi_1
# Внутренняя
r_3 = r_1 - d
phi_3 = phi_2

plt.polar(phi_1, r_1, 'ro', c='r')
plt.polar(phi_2, r_2, 'ro', c='g')
plt.polar(phi_3, r_3, 'ro', c='b')
plt.show()
########################
x_1, y_1 = toCartesean(r_1, phi_1)
x_2, y_2 = toCartesean(r_2, phi_2)
x_3, y_3 = toCartesean(r_3, phi_3)

print(np.sqrt((x_1-x_3)**2+(y_1-y_3)**2))
# TODO Для удобства перевести вводить координаты в полярной системе координат. Даллее перевести в декартовые и упорядочить!!!!
# append the starting x,y coordinates

xi, yi = plot_spline(x_1, y_1)
plt.plot(xi, yi, c='r')
xi, yi = plot_spline(x_2, y_2)
plt.plot(xi, yi, c='g')
xi, yi = plot_spline(x_3, y_3)
plt.plot(xi, yi, c='b')
plt.show()
