import matplotlib.pyplot as plt

# 自定义数据
x = [1, 2, 3, 4, 5, 6, 7]  # 横坐标 (编号)
acc1 = [30.33, 26.66, 32.13, 27.01, 29.94, 30.44, 30.35]  # acc1的成功率
acc5 = [82.25, 79.93, 82.76, 79.96, 82.18, 81.97, 82.12]  # acc5的成功率

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制 acc1 和 acc5 的曲线
plt.plot(x, acc1, label='acc1', color='b', marker='o')  # 蓝色，圆形标记
plt.plot(x, acc5, label='acc5', color='r', marker='s')  # 红色，方形标记

# 添加标题
plt.title('linprobe of different methods', fontsize=16)

# 添加横轴和纵轴标签
plt.xlabel('Methods', fontsize=12)
plt.ylabel('Success Rate', fontsize=12)

# 添加图例
plt.legend()

# 显示网格
plt.grid(True)

# 保存图像到指定路径
save_path = 'experiments/draw/methods_linprobe'  # 替换为你想保存的路径
plt.savefig(save_path)

# 显示图形
plt.show()
