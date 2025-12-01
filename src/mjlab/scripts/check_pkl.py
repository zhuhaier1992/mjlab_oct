import pickle
import numpy as np


pkl_file = "/home/robot/projects/motion_data/MimicKit_Data/motions/g1/g1_cartwheel.pkl"
# 加载文件
data = pickle.load(open("/home/robot/projects/motion_data/MimicKit_Data/motions/g1/g1_cartwheel.pkl", "rb"))
data.frames
data.fps
# 查看 key
print(data.keys())  # 输出所有 key（Python 终端会直接显示）
# print(data['frames'])
print(np.array(data['frames']).shape)
print(data['fps'])
print(data['loop_mode'])
# （可选）查看某个 key 的数据
# data["root_pos"].shape  # 查看 root_pos 的形状
# data["fps"]  # 查看帧率