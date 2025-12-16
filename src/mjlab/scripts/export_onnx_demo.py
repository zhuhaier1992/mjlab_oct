# export_onnx.py
import torch
import os
from pathlib import Path
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.rl.exporter import export_motion_policy_as_onnx, attach_onnx_metadata, attach_onnx_metadata_2
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import os
from dataclasses import asdict
from pathlib import Path
import gymnasium as gym
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from typing import Literal, cast
import tyro
from mjlab.viewer import NativeMujocoViewer, ViserViewer
from rsl_rl.runners import OnPolicyRunner

def export_from_checkpoint(task_name, checkpoint_path, output_dir, 
                           file_name="policy.onnx", motion_file_path=None):
    # env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    # agent_cfg = load_cfg_from_registry(task_name, "rl_cfg_entry_point")
    env_cfg = cast(
    ManagerBasedRlEnvCfg, load_cfg_from_registry(task_name, "env_cfg_entry_point"))
    agent_cfg = cast(
    RslRlOnPolicyRunnerCfg, load_cfg_from_registry(task_name, "rl_cfg_entry_point"))
    motion = False
    if motion_file_path:
        env_cfg.commands.motion.motion_file = motion_file_path
        motion = True

    env = gym.make(task_name, cfg=env_cfg, device="cpu", render_mode="rgb_array" )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    if motion:
        runner = MotionTrackingOnPolicyRunner(
            env, asdict(agent_cfg), log_dir=str(Path(checkpoint_path).parent), device="cpu"
        )
    else:
        runner = OnPolicyRunner(
            env, asdict(agent_cfg), log_dir=str(Path(checkpoint_path).parent),device="cpu"
        )

    runner.load(checkpoint_path, map_location="cpu")
    
    #################################################################
    # policy = runner.get_inference_policy(device="cpu")
    # NativeMujocoViewer(env, policy).run()
    
    # ################################################################
    # return
    
    if runner.alg.policy.actor_obs_normalization:
        normalizer = runner.alg.policy.actor_obs_normalizer
    else:
        normalizer = None
    
    export_motion_policy_as_onnx(
        env.unwrapped,
        runner.alg.policy,
        normalizer=normalizer,
        path=output_dir,
        filename=file_name,
        verbose=True,
        motion=motion
    )
    
    if motion:
        attach_onnx_metadata(env=env.unwrapped, run_path= "", 
                         path=output_dir, filename=file_name)
    else:
        attach_onnx_metadata_2(env.unwrapped, "", output_dir, file_name)
    print(f"ONNX模型已导出到: {output_dir}/{file_name}")

if __name__ == "__main__":
    export_from_checkpoint(
        task_name="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",  
        # task_name="Mjlab-Velocity-Rough-Unitree-G1",
        checkpoint_path="/home/robot/projects/mjlab/logs/rsl_rl/g1_tracking/2025-12-12_17-48-20/model_89999.pt",  # 检查点路径
        output_dir="/home/robot/projects/motion_data/onnx/",
        file_name="broadcast_1min.onnx",
        motion_file_path="/home/robot/projects/mjlab/artifacts/local/tu3_2.npz"
    )