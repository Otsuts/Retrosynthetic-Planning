import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from os import walk
import seaborn as sns


def draw_train():
    x = range(1, 21, 1)
    precision_1e_5 = [289.59,285.08,282.60,278.40,274.38,271.53,267.69,264.55,261.81,260.18,257.40,255.55,252.88,251.29,249.38,247.85,245.71,244.49,242.78,242.70]
    precision_5e_5 = [273.71,261.89,251.78,244.95,239.23,239.52,235.09,231.74,226.68,224.69,223.34,220.10,219.04,216.73,215.73,215.27,210.11,212.39,210.38,208.08]
    precision_1e_4 = [268.05,250.63,243.68,233.96,231.30,230.51,225.76,225.71,220.61,222.43,219.17,219.35,215.99,214.35,215.45,213.96,216.35,214.74,212.54,210.42]
    plt.plot(x, precision_1e_5,  label='LR1e-5')
    plt.plot(x, precision_5e_5, label='LR5e-5')
    plt.plot(x, precision_1e_4, label='LR1e-4')

    plt.title('Training loss change of different lr')
    plt.xlabel('Iterations')
    plt.ylabel('L2 loss')

    # 添加图例
    plt.legend()
    plt.savefig('task2_train.png')

def draw_test():
    x = range(1, 21, 1)
    precision_1e_5 = [741.03,740.09,750.37,749.43,748.45,745.21,744.99,744.98,741.85,739.04,740.97,733.89,736.21,732.07,731.66,733.18,736.46,729.74,736.67,726.26]
    precision_5e_5 = [741.75,729.82,726.21,717.88,719.76,713.91,712.18,708.38,721.69,721.52,702.04,702.31,697.94,723.87,691.86,683.77,699.31,698.05,708.64,708.66]
    precision_1e_4 = [728.49,724.03,720.43,719.09,721.50,698.22,706.84,696.77,715.77,722.92,711.92,724.06,714.97,719.29,721.45,712.40,726.84,727.26,721.39,719.98]
    plt.plot(x, precision_1e_5,  label='LR1e-5')
    plt.plot(x, precision_5e_5, label='LR5e-5')
    plt.plot(x, precision_1e_4, label='LR1e-4')

    plt.title('Testing loss change of different lr')
    plt.xlabel('Iterations')
    plt.ylabel('L2 loss')

    # 添加图例
    plt.legend()
    plt.savefig('task2_test.png')

def draw_bar():

    sns.set(color_codes=True)
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    # 柱高信息
    Y = [131,150, 153,26]
    Y1 = [21.98, 75.32,133.08,65.4]
    X = np.arange(len(Y))

    bar_width = 0.25
    tick_label = ['Topk=10','Topk=50','Topk=100','Ours']

    # 显示每个柱的具体高度
    for x, y in zip(X, Y):
        plt.text(x-0.15, y+0.005, '%.3f' % y, ha='center', va='bottom')

    for x, y1 in zip(X, Y1):
        plt.text(x+0.25, y1+0.005, '%.3f' % y1, ha='center', va='bottom')

    # 绘制柱状图
    plt.bar(X, Y, bar_width, align="center",
            color="red", label="Hit number", alpha=0.5)
    plt.bar(X+bar_width, Y1, bar_width, color="purple", align="center",
            label="Avg time cost", alpha=0.5)

    plt.xlabel("Topk values")
    plt.ylabel("Hit number and time cost")
    plt.title('Hit number and time cost with different topk')

    plt.xticks(X+bar_width/2, tick_label)
    # 显示图例
    plt.legend()
    # plt.show()
    plt.savefig('task3.png', dpi=400)


# draw_train()
# draw_test()
draw_bar()