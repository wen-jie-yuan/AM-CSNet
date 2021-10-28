# -*-coding:utf-8-*-
# ! /usr/bin/env python
"""
Author And Time : ywj 2021/10/8 15:10
Desc: paper2：在CPU上的重构质量和运行速度的比较。
比较的传统CS方法用蓝色字体标记，比较的基于深度学习的CS方法用绿色字体标记。该图基于Set11[20]结果，采样比为0.1。
"""
import matplotlib.pyplot as plt

# 参数配置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# 画布大小设置
plt.figure(figsize=(6, 6))
plt.xlabel(
    "Slower " + r'$\leftarrow$' + "                  Running time                  " + r'$\rightarrow$' + " Faster",
    fontsize=14,
    family='Times New Roman')
plt.ylabel("PSNR (dB)", fontsize=14, family='Times New Roman')  # style='italic'斜体
plt.tick_params(labelsize=14)
plt.xlim(1, 10 ** 3)
plt.ylim(23.85, 30.15)

# TV(25.47,10.84) MH(27.62,209.43) GSR(28.76,2535.3) D-AMP(29.21,403.4) /
# Reconnet(21.69, 5.7123) ISTA+(26.46,12.386) CSNet+(28.06,4.928) SCSNet(28.48,5.613) /
# AM-CSNet-8(29.23,2.959), AM-CSNet-8(29.58,7.161)
# traditional
x1 = [10.84, 209.43, 253.3, 403.4]
y1 = [25.47, 27.62, 28.76, 29.21]
name1 = ['$TV$', '$MH$', '$GSR$', '$D-AMP$']
# dl
x2 = [5.7123, 12.386, 4.928, 5.613]
y2 = [24.28, 26.46, 28.06, 28.48]
name2 = ['$Reconnet$', '$ISTA^+$', '$TIP-CSNet$', '$SCSNet$']
# am
x3 = [] #2.959
y3 = [] #29.23
name3 = ['$AM-CSNet-1$']
x4 = [7.161]
y4 = [29.61]
name4 = ['$AM-CSNet$']

plt.scatter(x1, y1, marker="o", color='b')
plt.scatter(x2, y2, marker="^", color='darkorange')
plt.scatter(x3, y3, marker="*", color='red')
plt.scatter(x4, y4, marker="*", color='red')
for i in range(len(x1)):
    plt.annotate(name1[i], xy=(x1[i], y1[i]), xytext=(x1[i] + 0.1, y1[i] + 0.1), fontsize=12, c='b')
for i in range(len(x2)):
    plt.annotate(name2[i], xy=(x2[i], y2[i]), xytext=(x2[i] + 0.1, y2[i] + 0.1), fontsize=12, c='darkorange')
for i in range(len(x3)):
    plt.annotate(name3[i], xy=(x3[i], y3[i]), xytext=(x3[i]+20, y3[i]-0.1), fontsize=12, c='red')
for i in range(len(x4)):
    plt.annotate(name4[i], xy=(x4[i], y4[i]), xytext=(x4[i], y4[i]+0.1), fontsize=12, c='red')
plt.xscale('log')

plt.gca().invert_xaxis()
# plt.grid(linestyle=":", color='r')
plt.tick_params(labelbottom='on', bottom=False)
plt.savefig(fname="./fig1.png", dpi=300)
plt.show()
