import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import subprocess
import os


def _function(x):
    a = 1000
    b = 10000
    return np.sqrt(x * x + (40000 - 1000) ** 2) - 40000 - 1000 + 2000


# return 500 * np.a(np.sin(x * 0.0005) + np.cos(x * 0.0009))
def _spline(x):
    pass


def _koefx(s, x0, derivate):
    return s * np.sqrt(1 / (1 + np.square(1 / derivate)))


def _coef_y(s, x0, derivate):
    return -1 * (1 / derivate) * _koefx(s, x0, derivate)


def get_function():
    return _function


# Решается задача оптимизации расстояния до кривой. В итоге получем нелинейное алг.уравнение относительно
# координаты x(длина нормали)
def distance_to_point(x0, y0):
    rho = lambda x: np.sqrt((x - x0) ** 2 + (_function(x) - y0) ** 2)
    _x = minimize(rho, x0).x
    return rho(_x)


class FunctionRoad:

    def __init__(self, s=400, start=0, end=40000, count_cone=100):
        self.s = s
        self.start = start
        self.end = end
        self.count_cone = count_cone
        self.xxc = None
        self.lcx = None
        self.lcy = None
        self.rcx = None
        self.rcy = None
        self.is_generated_data = False

    # Для поворота. Два уравнения.
    # Точки расположение на нормали и расстояние между ними s. Два решения - наши исходные точки

    def plot_cones(self, plot_func: bool = True, accuracy: int = 200):
        assert self.is_generated_data == True, "no data is generated"
        xx = np.linspace(self.start, self.end, accuracy)
        yy1 = _function(xx) + self.s
        yy2 = _function(xx) - self.s
        yy = _function(xx)
        if plot_func:
            plt.plot(xx, yy1, c='Black')
            plt.plot(xx, yy2, c='Black')
            plt.plot(xx, yy, c="Blue")

        plt.plot(self.lcx, self.lcy, 'o', c='Red')
        plt.plot(self.rcx, self.rcy, 'o', c='Red')
        plt.show()

    def generate_spline_data(start: np.array, end: np.array, n):
        pass

    def generate_data(self, output= True):
        # TODO Доделать потом
        # start = [2, 4]
        # z   end  = [4, 4]
        # В UnrealEngine4 в см
        # Конусы
        self.xxc = np.linspace(self.start, self.end, self.count_cone)[1::]
        deriv = derivative(_function, self.xxc)
        self.lcx = self.xxc + _koefx(self.s, self.xxc, deriv)
        self.lcy = _function(self.xxc) + _coef_y(self.s, self.xxc, deriv)
        self.rcx = self.xxc - _koefx(self.s, self.xxc, deriv)
        self.rcy = _function(self.xxc) - _coef_y(self.s, self.xxc, deriv)
        self.is_generated_data = True
        if output ==True:
            return np.array([[self.lcx,self.lcy],[self.rcx,self.rcy]]),np.array([self.xxc,_function(self.xxc)])
    def move_data(self):
        assert self.is_generated_data == True, "no data is generated"
        data_left_cone = pd.DataFrame({'X': self.lcx, 'Y': self.lcy})
        data_right_cone = pd.DataFrame({'X': self.rcx, 'Y': self.rcy})
        data_left_cone.to_csv("data_left_cone.csv")
        data_right_cone.to_csv("data_right_cone.csv")
        path_from = os.getcwd() + "\\*.csv "
        path_to = "D:\\Users\\user\\Documents\\Course\\MyProject2\\Content"
        cmd = "copy" + " " + path_from + " " + path_to + "/y"
        # print(cmd)
        returned_output = subprocess.check_output(cmd,
                                                  shell=True)  # returned_output содержит вывод в виде строки байтов

        print('Результат выполнения команды:', returned_output.decode("CP866"))  # Преобразуем байты в строку

    def reset_data(self):
        self.xxc = None
        self.lcx = None
        self.lcy = None
        self.rcx = None
        self.rcy = None
        self.is_generated_data = False


    def get_data(self):
        return self.xxc, self.lcx, self.lcy
