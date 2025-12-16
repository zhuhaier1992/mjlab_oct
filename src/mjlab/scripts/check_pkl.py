import pickle
import numpy as np


# pkl_file = "/home/robot/projects/motion_data/MimicKit_Data/motions/g1/g1_cartwheel.pkl"
# 加载文件
data = pickle.load(open("/home/robot/projects/motion_data/pkl/14008.pkl", "rb"))
# 查看 key
print(data.keys())  # 输出所有 key（Python 终端会直接显示）
print(data['fps'])
print(data['root_pos'].shape)
print(data['root_rot'].shape)
print(data['dof_pos'].shape)
print(data['local_body_pos'])
print(data['link_body_list'])