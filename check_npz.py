import numpy as np

file="/home/robot/projects/motion_data/CMU/01/01_01_stageii.npz"
data = np.load(file)
ks=data.keys()
print(list(ks))
print(data['mocap_frame_rate'])
print(data['pose_body'].shape)  # 输出 body_pos_w 的形状