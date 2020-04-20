import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dset1=pd.read_csv('C://Users//HP//usd_quotes.csv', sep=';')
#print(dset1.head(10))
#print(dset1.isnull().sum())
dset1 = dset1.dropna()
#print(dset1.isnull().sum())
dset = np.array(dset1['price'][::-1])
#print(dset)
print(len(dset))
x = list()
x1 = list()
for i in range(0,59):
    for j in range(0,30):
        x1.append(dset[i+j])
    x.append([x1, dset[i+j+1]])
    x1 = list()
for i in x:
    print(i)


# класс, который реализует персептрон и его обучение
class TPerceptron:
    def __init__(self, N):
        # создать нулевые веса
        self.w = list()
        for i in range(N):
            self.w.append(0)
    # метод для вычисления значения персептрона
    def calc(self, x):
        res = 0
        for i in range(len(self.w)):
            res = res + self.w[i] * x[i]
        return res
    # обучение на одном примере
    def learn(self, la, x, y):
        # обучаем только, когда результат неверный
        if y * self.calc(x) <=  0:
            for i in range(len(self.w)):
                self.w[i] = self.w[i] + la * y * x[i]
    # обучение по всем данным T - кортеж примеров
    def learning(self, la, T):
        # цикл обучения
        for n in range(100):
            # обучение по всем набору примеров
            for t in T:
                self.learn(la, t[0], t[1])
# создаем класс двумерного персептрона
'''perceptron = TPerceptron(3)
la =  0.1 # константа обучения
# создаем примеры
T = list()
T.append([[2,1,3], 1])
T.append([[3,2,3], 1])
T.append([[4,1,3], 1])
T.append([[1,2,3], -1])
T.append([[2,3,3], -1])
T.append([[5,7,3], -1])
print(T)
perceptron.learning(la, T) # обучение персептрона
print(perceptron.w) # печатаем веса
# проверим работу на тестовых примерах
print(perceptron.sign([1.5, 2,4]))
print(perceptron.sign([3, 1.5,4]))
print(perceptron.sign([5,1,4]))
print(perceptron.sign([5,10,4]))'''

perceptron1 = TPerceptron(30)
la = 0.00001
perceptron1.learning(la,x) # обучение персептрона
print("Веса ",perceptron1.w) # печатаем веса
# проверим работу на тестовых примерах
print('Результат ',perceptron1.calc(x[58][0]))
print(x[58][0])