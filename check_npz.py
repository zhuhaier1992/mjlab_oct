import numpy as np

file="/home/robot/projects/mjlab/artifacts/tu:v0/motion.npz"
data = np.load(file)
print(data.keys())
print(data['fps'])
print(data['body_pos_w'].shape)  # 输出 body_pos_w 的形状