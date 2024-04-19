import matplotlib.pyplot as plt

# 数据
epochs = [0, 1, 2, 3, 4, 5]
AUROC = [0.844, 0.876, 0.883, 0.881, 0.883, 0.882]
accuracy = [0.711, 0.735, 0.732, 0.734, 0.739, 0.735]

# 创建画布和轴对象
fig, ax1 = plt.subplots()

# 绘制AUROC折线图
ax1.plot(epochs, AUROC, marker='o', color='blue', label='AUROC')
ax1.set_ylabel('AUROC')
ax1.tick_params(axis='y')
ax1.set_ylim(0.84, 0.9)

# 创建第二个纵轴对象
ax2 = ax1.twinx()

# 绘制Accuracy折线图
ax2.plot(epochs, accuracy, marker='o', color='red', label='Accuracy')
ax2.set_ylabel('Accuracy')
ax2.tick_params(axis='y')
ax2.set_ylim(0.7, 0.75)

# 设置图表标题和横轴标签
ax1.set_xlabel('Epochs')

# 同时显示图例
lines = [ax1.get_lines()[0], ax2.get_lines()[0]]
plt.legend(lines, [line.get_label() for line in lines])

# 展示图表
fig.set_size_inches(8, 4)
# plt.show()
plt.savefig('auroc.png')