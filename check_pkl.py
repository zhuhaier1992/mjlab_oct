import pickle
import numpy as np

# 替换为你的 pkl 文件路径
pkl_path = "/home/robot/projects/motion_data/pkl/amass0101.pkl"

# 加载 pkl 文件
with open(pkl_path, "rb") as f:
    motion_data = pickle.load(f)

# 1. 查看所有键（确认包含的字段）
print("所有字段：", motion_data.keys())
# 输出应为：dict_keys(['fps', 'root_pos', 'root_rot', 'dof_pos', 'local_body_pos', 'link_body_list'])

# 2. 逐个查看字段内容
# 查看帧率
print("\nfps：", motion_data["fps"])