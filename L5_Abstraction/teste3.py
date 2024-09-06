import numpy as np
from datetime import datetime
# import tensorflow as tf
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# print("Num GPUs Available: ", 
#     len(tf.config.experimental.list_physical_devices('GPU')))
time_format = "%d-%m-%Y;%H:%M:%S"
a = [[1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,99],
    [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,98],
    [31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,97]]

a = np.array(a)
a = 2*a
print(a)
x = [1,2,3]
y = [4,5,6]
print(x + y)
print()
print()
print()

print(("Temperature: {} °C\n" +
        "Humidity: {}%\n" +
        "Visible: {} lm | IR: {} lm | UV Index: {}\n" +
        "Control Mode: {}\n" +
        "Pump State: {} | Light State: {}\n").format(
            22.4,
            87.6,
            478,
            1158,
            0.9,
            1,
            0,
            0
        ))
# print(a[:, 0:12])
# print(a[:, 12])
# print(a[:i1])
# print(a[i1:i2])
# print(a[:i2])


# b = a[:, 14]
# c = a[:, [0,6]]

# j = [1,2,3]
# k = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15])
# print(k[[0,6]])

# a = [1,2,3]
# b = [4,5,6]
# c = []
# c.append(a)
# c.append(b)

# print(c)


# data = [["28-08-2024","17:07:49",23.8,92.0,263,264,0.0,0,0,0],
#         ["28-08-2024","17:07:50",24.0,92.0,264,265,0.0,0,0,0],
#         ["28-08-2024","17:07:51",23.9,92.0,263,265,0.0,0,0,0],
#         ["28-08-2024","17:07:53",24.0,92.0,264,265,0.0,0,0,0],
#         ["28-08-2024","17:07:54",24.0,92.0,264,265,0.0,0,0,0],
#         ["28-08-2024","17:07:55",23.8,92.0,263,265,0.0,0,0,0],
#         ["28-08-2024","17:07:56",23.9,92.0,265,264,0.0,0,0,0],
#         ["28-08-2024","17:07:57",24.1,92.0,263,263,0.0,0,0,0],
#         ["28-08-2024","17:08:00",24.1,92.0,262,265,0.0,0,0,0],
#         ["28-08-2024","17:08:01",23.7,92.0,263,263,0.0,0,0,0],
#         ["28-08-2024","17:08:02",24.0,92.0,261,264,0.0,0,0,0],
#         ["28-08-2024","17:08:03",24.2,92.0,264,264,0.0,0,0,0],
#         ["28-08-2024","17:08:04",26.5,92.0,262,262,0.0,0,0,1],
#         ["28-08-2024","17:08:05",26.5,92.0,338,641,0.4,0,0,1],
#         ["28-08-2024","17:08:07",26.1,92.0,337,640,0.4,0,0,1],
#         ["28-08-2024","17:08:08",26.5,92.0,338,642,0.4,0,0,1],
#         ["28-08-2024","17:08:09",26.4,92.0,336,642,0.4,0,0,1],
#         ["28-08-2024","17:08:10",28.0,92.0,336,640,0.4,0,1,1],
#         ["28-08-2024","17:08:11",28.0,92.0,337,641,0.4,0,1,1],
#         ["28-08-2024","17:08:12",28.0,92.0,337,637,0.4,0,1,1],
#         ["28-08-2024","17:08:14",28.0,92.0,337,643,0.4,0,1,1],
#         ["28-08-2024","17:08:15",28.0,92.0,338,640,0.4,0,1,1],
#         ["28-08-2024","17:08:16",28.0,92.0,337,640,0.4,0,1,1],
#         ["28-08-2024","17:08:17",28.0,92.0,338,641,0.4,0,1,1],
#         ["28-08-2024","17:08:18",28.0,92.0,337,638,0.4,0,1,1],
#         ["28-08-2024","17:08:19",28.0,92.0,338,641,0.4,0,1,1],
#         ["28-08-2024","17:08:20",28.0,92.0,337,639,0.4,0,1,1],
#         ["28-08-2024","17:08:22",26.5,92.0,339,638,0.4,0,0,1],
#         ["28-08-2024","17:08:23",26.4,92.0,339,641,0.4,0,0,1],
#         ["28-08-2024","17:08:24",25.9,92.0,337,638,0.4,0,0,1],
#         ["28-08-2024","17:08:25",26.3,92.0,339,642,0.4,0,0,1],
#         ["28-08-2024","17:08:26",26.3,92.0,338,638,0.4,0,0,1],
#         ["28-08-2024","17:08:27",26.3,92.0,338,642,0.4,0,0,1],
#         ["28-08-2024","17:08:29",26.2,92.0,337,639,0.4,0,0,1],
#         ["28-08-2024","17:08:30",25.7,92.0,337,640,0.4,0,0,1],
#         ["28-08-2024","17:08:31",26.2,92.0,339,639,0.4,0,0,1],
#         ["28-08-2024","17:08:32",26.2,92.0,337,639,0.4,0,0,1],
#         ["28-08-2024","17:08:33",26.2,92.0,336,640,0.4,0,0,1],
#         ["28-08-2024","17:08:34",26.2,92.0,337,638,0.4,0,0,1],
#         ["28-08-2024","17:08:36",25.8,92.0,339,641,0.4,0,0,1],
#         ["28-08-2024","17:08:37",26.2,92.0,338,639,0.4,0,0,1],
#         ["28-08-2024","17:08:38",26.3,92.0,337,641,0.4,0,0,1],
#         ["28-08-2024","17:08:39",26.2,92.0,338,641,0.4,0,0,1],
#         ["28-08-2024","17:08:40",26.1,92.0,337,638,0.4,0,0,1],
#         ["28-08-2024","17:08:41",25.8,92.0,338,641,0.4,0,0,1],
#         ["28-08-2024","17:08:43",26.1,92.0,336,640,0.4,0,0,1],
#         ["28-08-2024","17:08:45",24.0,92.0,264,264,0.0,1,0,0],
#         ["28-08-2024","17:08:47",26.0,92.0,336,633,0.4,1,0,1],
#         ["28-08-2024","17:08:48",26.4,92.0,339,639,0.4,1,0,1],
#         ["28-08-2024","17:08:49",26.4,92.0,338,640,0.4,1,0,1],
#         ["28-08-2024","17:08:51",26.3,92.0,338,640,0.4,1,0,1],
#         ["28-08-2024","17:08:52",26.3,92.0,338,642,0.4,1,0,1],
#         ["28-08-2024","17:08:53",25.8,92.0,337,640,0.4,1,0,1],
#         ["28-08-2024","17:08:54",26.4,92.0,337,641,0.4,1,0,1],
#         ["28-08-2024","17:08:55",26.3,92.0,337,638,0.4,1,0,1],
#         ["28-08-2024","17:08:56",26.3,92.0,337,639,0.4,1,0,1]]

# last_value = -1
# ini_index = []
# fin_index = []
# counter = 0
# counters = []
# for i in range(len(data)):
#     current_value = data[i][8]
#     if(current_value == 0): counter+=1
#     if(current_value != last_value and current_value == 0):
#         ini_index.append(i)

#     if(current_value != last_value and current_value == 1):
#         counters.append(counter)
#         counter = 0
#         fin_index.append(i-1)

#     if((i == (len(data)-1)) and current_value == 0):
#         counters.append(counter)
#         counter = 0
#         fin_index.append(i)

#     last_value = current_value

# print(ini_index)
# print(fin_index)
# elapsed_interval    = []
# elapsed_activation  = []
# X = [-1 for i in range(len(data))]
# Y = [-1 for i in range(len(data))]
# print(len(X))
# print(len(Y))
# for i in (range(len(ini_index)-1)):
#     interval_ini_time    = datetime.strptime(str(data[ini_index[i]][0] + ';' + data[ini_index[i]][1])    ,time_format)   # Merges separated date and hour presented in time_format format into a datetime instance
#     interval_fin_time    = datetime.strptime(str(data[fin_index[i]][0] + ';' + data[fin_index[i]][1])    ,time_format) 
#     elapsed_interval.append((interval_fin_time-interval_ini_time).total_seconds())

#     activation_ini_time    = datetime.strptime(str(data[fin_index[i]+1][0] + ';' + data[fin_index[i]+1][1])    ,time_format)   # Merges separated date and hour presented in time_format format into a datetime instance
#     activation_fin_time    = datetime.strptime(str(data[ini_index[i+1]-1][0] + ';' + data[ini_index[i+1]-1][1])    ,time_format) 
#     elapsed_activation.append((activation_fin_time-activation_ini_time).total_seconds())

#     for j in range(ini_index[i+1] - ini_index[i]):
#         X[j] = (elapsed_interval[i])
#         Y[j] = (elapsed_activation[i])
#     # interval    = len(range(ini_index[i], fin_index[i]))+1
#     # activation  = len(range(fin_index[i], ini_index[i+1]))-1

# print(elapsed_interval)
# print(elapsed_activation)

# print(X)
# print(Y)


# Temperatura   ,Umidade relativa,Intensidade de luz no espectro visível,Intensidade de luz no espectro infravermeho,Índice UV,Tempo decorrido desde o início do cultivoIntervalo de tempo entre dois recebimentos de dadosTaxa de variação da temperatura   Taxa de variação da umidade relativa, taxa de variação da intensidade de luz no espectro visível, taxa de variação da intensidade de luz no espectro infravermeho, taxa de variação do indice UV