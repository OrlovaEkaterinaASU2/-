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
f = open('real.txt', 'w')
for i in range(0,59):
    for j in range(0,30):
        x1.append(dset[i+j])
        f.write(str(dset[i+j]) + '\n')
    x.append([x1, dset[i+j+1]])
    x1 = list()
for i in x:
    print(i)
f.close()

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
# создаем класс  персептрона
perceptron1 = TPerceptron(30)
la = 0.00001
x1 = x[0:30]
perceptron1.learning(la,x1) # обучение персептрона
print("Веса ",perceptron1.w) # печатаем веса
# проверим работу на тестовых примерах
print('Результат ', perceptron1.calc(x[30][0]))
print(x[58][0])

f1 = open('test1.txt', 'w')
for i in range(0,30):
    f1.write(str(perceptron1.calc(x[i][0])) + '\n')
f1.close()

f2 = open('test2.txt', 'w')
for i in range(30,59):
    f2.write(str(perceptron1.calc(x[i][0])) + '\n')
f2.close()