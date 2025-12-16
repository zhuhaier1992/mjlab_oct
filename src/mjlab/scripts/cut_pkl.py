import pickle
import math

src_pkl = "/home/robot/projects/motion_data/pkl/14008.pkl"      # 原始文件
dst_pkl = "/home/robot/projects/motion_data/pkl/14008_cut.pkl"  # 输出文件

# 1. 读原数据
with open(src_pkl, "rb") as f:
    data = pickle.load(f)

fps   = data["fps"]          # 30 fps 左右
n_all = data["root_pos"].shape[0]  # 223
dur_all = n_all / fps        # 7.45 s

# 2. 计算要剪掉的帧数（最后 2 秒）
trim = int(math.ceil(2.0 * fps))  # 向上取整，避免多留
n_new = n_all - trim
assert n_new > 0, "剪得太狠，没有剩余帧"

# 3. 切片
def _cut(arr):
    if arr is None:
        return None
    return arr[:n_new]

data_new = {
    "fps": data["fps"],  # fps 不变
    "root_pos": _cut(data["root_pos"]),      # (N,3)
    "root_rot": _cut(data["root_rot"]),      # (N,4)
    "dof_pos":  _cut(data["dof_pos"]),       # (N,29)
    "local_body_pos": _cut(data["local_body_pos"]),
    "link_body_list": data["link_body_list"]  # 非时序，保持原样
}

# 4. 另存
with open(dst_pkl, "wb") as f:
    pickle.dump(data_new, f)

print(f"已保存 {dst_pkl}")
print(f"原帧数：{n_all}，新帧数：{n_new}，剪掉了 {trim} 帧")