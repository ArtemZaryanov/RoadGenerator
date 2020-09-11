import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
import os

def function(x: float): return np.cos(0.1 * x) + np.sin(0.01 * x)


s = 4
# TODO Доделать потом
# start = [2, 4]
# end  = [4, 4]
start = 0
end = 80
count_cone = 20
xx = np.linspace(start,end, 200)
yy1 = function(xx) + s
yy2 = function(xx) - s
lcx = np.linspace(start,end, count_cone)
rcx = np.linspace(start,end, count_cone)
lcy = function(lcx) + s
yy = function(xx)
rcy = function(rcx) - s
plt.plot(xx,yy1, c='Black')
plt.plot(xx,yy2, c='Black')
plt.plot(lcx,lcy,'o', c='Red')
plt.plot(rcx,rcy,'o', c='Red')
plt.plot(xx,yy, c= "Blue")
plt.show()

data_left_cone = pd.DataFrame({'X': lcx, 'Y': lcy})
data_right_cone = pd.DataFrame({'X': rcx, 'Y': rcy})
print(data_left_cone)
print(data_right_cone)
data_left_cone.to_csv("data_left_cone.csv")
data_right_cone.to_csv("data_right_cone.csv")
path_from = os.getcwd() + "\\*.csv "
path_to = "D:\\Users\\user\\Documents\\Course\\MyProject2\\Content"
cmd = "copy" + " " + path_from + " " + path_to + "/y"
print(cmd)
returned_output = subprocess.check_output(cmd,shell=True) # returned_output содержит вывод в виде строки байтов

print('Результат выполнения команды:', returned_output.decode("CP866")) # Преобразуем байты в строку
