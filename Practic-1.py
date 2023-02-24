import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time


buffer = []

with open('examp11.txt') as f:
    for i in f:
        buffer.append(i)

with open('examp11_1.txt', 'w') as f:
    for item in buffer:
        f.write(item.replace(';', ','))


data = pd.read_csv('examp11_1.txt', header = None)

data.head()

lidar = data.values.tolist() # перевод dataframe в массив

angle = [] # хранятся углы

for i in np.arange(-120, 120, 240 / 681):
    angle.append(i)

arr = np.radians(np.array(angle))  # переводим в радиана т.к работаем в радианных значения.

x = []
y = []
xx = []
yy = []
plt.ion()
for st in range(len(lidar)):
    xx.append(lidar[st][0])
    yy.append(lidar[st][1])
    plt.clf()
    plt.scatter(x,y,c = "green", s = 7)
    plt.scatter(xx,yy,c = "orange", s = 7)
    plt.draw()
    plt.gcf().canvas.flush_events()
    time.sleep(0.001)
    for i in range(3, 684):
        if 1.3 < lidar[st][i] < 4.8:
              x.append(lidar[st][0] + 0.3 * math.cos(lidar[st][2]) + lidar[st][i] * math.cos(lidar[st][2] - arr[i - 3]))
              y.append(lidar[st][1] + 0.3 * math.sin(lidar[st][2]) + lidar[st][i] * math.sin(lidar[st][2] - arr[i - 3]))

plt.ioff()
plt.show()