from common.evaluation import straightness
from imagedata.synthetic.gendata import genLines, genNormLine
from imagedata.synthetic.imaging import distort
from proposed.model.samples import PlumbLine
from proposed.model.camera_model import CameraModel

import numpy as np
import matplotlib.pyplot as plt

import time
import pickle
plt.rcParams["figure.figsize"] = (8, 8)

class TestCase():
    def __init__(self, scale=2000, cod=(0, 0), degree=3, homo=True, k=-0.04, axis=0, tilt=0.1, alpha=1.0001, gamma=0.0001, discrete=False):
        self.cm = CameraModel(scale, cod, degree, homo)
        self.scale = scale
        self.k = k
        self.axis = axis
        self.tilt = tilt
        self.alpha = alpha
        self.gamma = gamma
        self.discrete = discrete
        # fixed test set
        self.test_lines = genLines(10, 0.3)
        self.test_lines.extend(genLines(10))
        self.test_pls = self.build_plumblines(self.test_lines)

    def build_plumblines(self, lines):
        pls:list[PlumbLine] = list()
        for line in lines:
            dist = distort(line, self.k, self.axis, self.tilt, self.alpha, self.gamma, self.scale, discrete=self.discrete)
            plt.scatter(dist[:, 0], dist[:, 1])
            pls.append(PlumbLine(dist, self.cm))
        plt.show()
        return pls
    
    def set_trainset(self, lines):
        self.train_lines = lines
        self.train_pls = self.build_plumblines(self.train_lines)
        self.cm.hlr = 1 / len(self.train_lines)
        self.cm.clr = 200 / len(self.train_lines)
        print(self.cm.clr)

    def train(self, iter=400):
        t1 = time.time()
        self.cm.estimate(self.train_pls, iters=iter)
        t2 = time.time()
        print("execution time:", t2-t1)
        print(self.cm.cod, self.cm.dist_coeff)
        print(self.cm.homography)
        fig, axs = plt.subplots(3, 1)
        cod_list = np.array(self.cm.cod_list)
        axs[0].plot(cod_list[:, 0], label='cx')
        axs[0].legend()
        axs[1].plot(cod_list[:, 1], label='cy')
        axs[1].legend()
        axs[2].plot(self.cm.residual, label='train error')
        axs[2].legend()
        # d = dict()
        # d['cod'] = cod_list
        # d['res'] = self.cm.residual
        # f = open("d1" + '.pkl', "wb")
        # import pickle
        # pickle.dump(d, f)
        self.train_error = self.cm.residual[-1]
        print("average train straightness:", self.train_error)
        plt.show()
    
    def evaluate(self):
        errors = list()
        for i in range(len(self.test_lines)):
            line = self.test_lines[i]
            pl = self.test_pls[i]
            pl.update_model()
            unline = pl.restore_image()
            plt.scatter(pl.origin[:, 0], pl.origin[:, 1], color='blue', label='distorted')
            plt.scatter(unline[:, 0], unline[:, 1], color='red', label='corrected')
            errors.append(straightness(unline))
        self.test_error = np.mean(np.array(errors))
        print("average test straightness: ", self.test_error)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=32)
        plt.show()
    

def test1():
    # various norm vector
    tc = TestCase()
    train_lines = genLines(8)
    train_errors = list()
    test_errors = list()
    for i in range(8):
        lines = train_lines[i: i+1]
        tc.set_trainset(lines)
        tc.train()
        tc.evaluate()
        train_errors.append(tc.train_error)
        test_errors.append(tc.test_error)
    plt.plot(train_errors, label='train error')
    plt.plot(test_errors, label='test error')
    plt.legend()
    plt.show()

def test2():
    # various distance to center
    tc = TestCase()
    # train_lines = [genNormLine(0, i/10) for i in range(1, 10)]
    train_errors = list()
    test_errors = list()
    for i in range(1, 11):
        lines = genLines(1, i/10)
        tc.set_trainset(lines)
        tc.train(iter=800)
        tc.evaluate()
        train_errors.append(tc.train_error)
        test_errors.append(tc.test_error)
    plt.plot(train_errors, label='train error')
    plt.plot(test_errors, label='test error')
    plt.legend()
    plt.show()

def test3():
    # various order
    tc = TestCase(k=-0.15)
    # tc = TestCase()
    train_lines = genLines(4, 1)
    train_lines.extend(genLines(4, 0.3))
    train_errors = list()
    test_errors = list()
    for i in range(1, 7):
        tc.cm.degree=i
        tc.set_trainset(train_lines)
        tc.train()
        tc.evaluate()
        train_errors.append(tc.train_error)
        test_errors.append(tc.test_error)
    plt.plot(train_errors, label='train error')
    plt.plot(test_errors, label='test error')
    f = open('order.pkl', 'wb')
    pickle.dump(test_errors, f)
    plt.legend()
    plt.show()

def test4():
    # fixing overfitting
    tc = TestCase()
    train_lines = [genNormLine(0, 1), genNormLine(0, 0.5)]
    train_errors = list()
    test_errors = list()
    for i in range(1, 12):
        tc.cm.degree=i
        tc.set_trainset(train_lines)
        tc.train()
        tc.evaluate()
        train_errors.append(tc.train_error)
        test_errors.append(tc.test_error)
    plt.plot(train_errors)
    plt.plot(test_errors)
    plt.show()

def test5():
    # number of lines
    train_errors = list()
    test_errors = list()
    for i in range(1, 17):
        tc = TestCase(cod=(50, 50))
        train_lines = genLines(i)
        tc.set_trainset(train_lines)
        tc.train()
        tc.evaluate()
        train_errors.append(tc.train_error)
        test_errors.append(tc.test_error)
    plt.plot(train_errors, label='train error')
    plt.plot(test_errors, label='test error')
    plt.legend()
    plt.show()

def test6():
    #convergence time
    tc = TestCase(cod=(50, 50))
    train_lines = genLines(4)
    tc.set_trainset(train_lines[:2])
    # train_lines = genLines(16)
    # tc.set_trainset(train_lines)
    tc.train()
    tc.evaluate()

def test7():
    #convergence time
    tc = TestCase(cod=(50, 50))
    train_lines = genLines(4)
    train_lines.extend(genLines(4, 0.3))
    tc.set_trainset(train_lines)

    tc.train(iter=300)
    tc.evaluate()

def dp_test():
    num_line = 4
    lines = genLines(num_line, crop=True)

    # lines.extend(genLines(num_line, 0.5, crop=False))
    pls:list[PlumbLine] = list()
    scale = 2000
    k = -0.04
    axis = 0
    tilt = 0.08
    alpha = 1.0001
    gamma = 0.0001
    cm = CameraModel(scale, (0, 0), 3, homo=True, lr=1/num_line)
    for line in lines:
        dist = distort(line, k, axis, tilt, alpha, gamma, scale, discrete=False)
        plt.scatter(dist[:, 0], dist[:, 1])
        pls.append(PlumbLine(dist, cm))
    plt.show()
    import time
    t1 = time.time()
    cm.estimate(pls, iters=400)
    t2 = time.time()
    print("execution time:", t2-t1)
    # the cod convergence
    cod_list = np.array(cm.cod_list)
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(cod_list[:, 0])
    axs[1].plot(cod_list[:, 1])
    axs[2].plot(cm.residual)
    plt.show()


    print(cm.cod, cm.dist_coeff)
    print(cm.homography)
    # training error
    res_list = list()
    for lin in pls:
        res_list.append(lin.avg_res*scale**2)
    # plt.scatter(np.arange(num_line) * 2 * np.pi / num_line, res_list)
    # plt.plot(res_list)
    # plt.show()
    print("mean residual: ", np.sqrt(np.mean(np.array(res_list))))
    
    # test error
    test_lines = genLines(10, 0.3)
    test_lines.extend(genLines(10))
    st_list = list()
    for lin in test_lines:
        dist = distort(lin, k, axis, tilt, alpha, gamma, scale, discrete=False)
        pl = PlumbLine(dist, cm)
        pl.update_model()
        unline = pl.restore_image()
    #     # print("straightness: ", pl.straightness())
        st_list.append(straightness(unline))
        plt.scatter(lin[:, 0], lin[:, 1], color='blue')
        plt.scatter(unline[:, 0], unline[:, 1], color='red')
    print("average straightness: ", np.sqrt(np.mean(np.array(st_list))))
    plt.show()
    # plt.scatter(np.arange(20) * 2 * np.pi / 20, st_list)
    # plt.show()


if __name__ == "__main__":
    # dp_test()
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
    test7()
