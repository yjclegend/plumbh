import matplotlib.pyplot as plt
import pickle
import numpy as np

from imagedata.synthetic.gendata import genLines, genNormLine
from imagedata.synthetic.imaging import distort
# plt.style.use('seaborn')

def figure1():
    f1 = open('d2.pkl', 'rb')
    d1 = pickle.load(f1)

    f2 = open('d16.pkl', 'rb')
    d2 = pickle.load(f2)

    fig, axs = plt.subplots(3, 1)

    axs[0].set_title("cx")
    axs[0].plot(d1['cod'][:, 0], label='2 lines')
    axs[0].plot(d2['cod'][:, 0], label='16 lines')
    axs[0].legend()
    axs[1].set_title("cy")
    axs[1].plot(d1['cod'][:, 1], label='2 lines')
    axs[1].plot(d2['cod'][:, 1], label='16 lines')
    axs[1].legend()
    axs[2].set_title("training error")
    axs[2].plot(d1['res'], label='2 lines')
    axs[2].plot(d2['res'], label='16 lines')
    axs[2].legend()

    plt.show()

def figure2():
    train_lines = genLines(8)
    for i in range(8):
        line = train_lines[i]
        plt.scatter(line[:, 0], line[:, 1], label='line_' + str(i))
    plt.legend(fontsize=8)
    plt.show()

def figure3():
    plt.xlim(-1000, 2000)
    for i in range(1, 11):
        line = genNormLine(0, i/10)
        dist = distort(line, -0.04, 0, 0.1, 1.0001, 0.0001, 2000, discrete=False)
        plt.scatter(dist[:, 0], dist[:, 1], label='line_' + str(i))
    plt.legend(fontsize=8)
    plt.show()

def figure4():
    f1 = open('order010.pkl', 'rb')
    d1 = pickle.load(f1)

    f2 = open('order004.pkl', 'rb')
    d2 = pickle.load(f2)
    f3 = open('order015.pkl', 'rb')
    d3 = pickle.load(f3)

    plt.style.use('seaborn')
    plt.plot(d1, label='k=-0.1')
    plt.plot(d2, label='k=-0.04')
    plt.plot(d3, label='k=-0.15')
    plt.xticks(np.arange(1, 7))
    plt.legend()
    plt.show()

def figure5():
    plt.style.use('seaborn')
    plt.plot()
figure5()