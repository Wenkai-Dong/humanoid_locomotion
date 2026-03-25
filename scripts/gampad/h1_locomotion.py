# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates an interactive demo with the H1 rough terrain environment.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/h1_locomotion.py

"""

"""Launch Isaac Sim Simulator first."""
import importlib.metadata as metadata
installed_version = metadata.version("rsl-rl-lib")
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import scripts.rsl_rl.cli_args as cli_args  # isort: skip


from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates an interactive demo with the H1 rough terrain environment."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
# Se2Gamepad
from isaaclab.devices import Se2Gamepad, Se2GamepadCfg

import torch
from rsl_rl.runners import OnPolicyRunner

import carb
import omni
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY
from humanoid_locomotion.tasks.velocity.dual_gate.config.h1.teacher_env_cfg import H1TeacherEnvCfg_PLAY

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

HEATMAP_MARKERS = {}
for i in range(10):
    # 计算颜色: 0=蓝 -> 9=红
    ratio = i / 9.0
    color = (ratio, 0.0, 1.0 - ratio)
    # 定义球体样式
    HEATMAP_MARKERS[f"level_{i}"] = sim_utils.SphereCfg(
        radius=0.02,  # 球的大小
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    )
MY_HEATMAP_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/AttentionHeatmap",  # 在USD里的路径
    markers=HEATMAP_MARKERS,  # 塞入我们生成的10个球
)
TASK = "Velocity-DualGate-Teacher-H1-Play-v0"
RL_LIBRARY = "rsl_rl"


class H1RoughDemo:
    """This class provides an interactive demo for the H1 rough terrain environment.
    It loads a pre-trained checkpoint for the Isaac-Velocity-Rough-H1-v0 task, trained with RSL RL
    and defines a set of keyboard commands for directing motion of selected robots.

    A robot can be selected from the scene through a mouse click. Once selected, the following
    keyboard controls can be used to control the robot:

    * UP: go forward
    * LEFT: turn left
    * RIGHT: turn right
    * DOWN: stop
    * C: switch between third-person and perspective views
    * ESC: exit current third-person view"""

    def __init__(self):
        """Initializes environment config designed for the interactive model and sets up the environment,
        loads pre-trained checkpoints, and registers keyboard events."""
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        # load the trained jit policy
        # checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
        checkpoint = "D:/humanoid_locomotion/logs/rsl_rl/dualgate_teacher_h1_v0/2026-03-22_15-27-07/model_19400.pt"
        # create envionrment
        env_cfg = H1TeacherEnvCfg_PLAY()
        env_cfg.scene.num_envs = 32
        env_cfg.episode_length_s = 1000000
        env_cfg.curriculum = None
        # env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        # env_cfg.commands.base_velocity.ranges.heading = (-1.0, 1.0)
        # wrap around environment for rsl-rl
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        # load previously trained model
        ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        ppo_runner.load(checkpoint)
        # obtain the trained policy for inference
        self.policy = ppo_runner.get_inference_policy(device=self.device)

        self.create_camera()

        self.gamepad = Se2Gamepad(Se2GamepadCfg(
            v_x_sensitivity=1.5,
            v_y_sensitivity=1.,
            omega_z_sensitivity=1.,
            dead_zone=0.01,
        ))
        self.gamepad.add_callback(carb.input.GamepadInput.Y, self._toggle_camera_cb)
        self.gamepad.add_callback(carb.input.GamepadInput.X, self._deselect_robot_cb)

        self.commands = torch.zeros(env_cfg.scene.num_envs, 3, device=self.device)
        self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")
        # self.set_up_keyboard()
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        self._camera_local_transform = torch.tensor([-2.8, 0.0, 0.8], device=self.device)

    def create_camera(self):
        """Creates a camera to be used for third-person view."""
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        # Create camera
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)

    def _toggle_camera_cb(self):
        if self._selected_id is not None:
            if self.viewport.get_active_camera() == self.camera_path:
                self.viewport.set_active_camera(self.perspective_path)
            else:
                self.viewport.set_active_camera(self.camera_path)

    def _deselect_robot_cb(self):
        self._prim_selection.clear_selected_prim_paths()

    def update_selected_object(self):
        """Determines which robot is currently selected and whether it is a valid H1 robot.
        For valid robots, we enter the third-person view for that robot.
        When a new robot is selected, we reset the command of the previously selected
        to continue random commands."""

        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            # a valid robot was selected, update the camera to go into third-person view
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3][0:4] == "env_":
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("The selected prim was not a H1 robot")

        # Reset commands for previously selected robot if a new one is selected
        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")

    def _update_camera(self):
        """Updates the per-frame transform of the third-person view camera to follow
        the selected robot's torso transform."""

        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[self._selected_id, :]  # - env.scene.env_origins
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id, :]

        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)

    def process_input(self):
        vel = self.gamepad.advance().to(self.device)
        print(vel)

        if self._selected_id is not None:
            # H1 的指令格式通常是 [x_vel, y_vel, z_vel, yaw_vel]
            # 对应的索引是 0, 1, 2, 3
            self.commands[self._selected_id, 0] = vel[0]  # 前进速度
            self.commands[self._selected_id, 1] = -vel[1]  # 横移速度
            self.commands[self._selected_id, 2] = -vel[2]  # 转向速度(Yaw)



def main():
    """Main function."""
    demo_h1 = H1RoughDemo()
    obs, _ = demo_h1.env.reset()
    heatmap_visualizer = VisualizationMarkers(MY_HEATMAP_CFG)
    scanner = demo_h1.env.unwrapped.scene["critic_height_scanner"]
    COLOR_SCALE = 50.0
    while simulation_app.is_running():
        # check for selected robots
        demo_h1.update_selected_object()
        demo_h1.process_input()
        demo_h1.policy.mha.mhas[0].need_weights = True
        with torch.inference_mode():
            action = demo_h1.policy(obs)
            attn_weights = demo_h1.policy.mha.mhas[0].attn_weights
            obs, _, _, _ = demo_h1.env.step(action)
            # overwrite command based on keyboard input
            obs["actor"][:, 9:12] = demo_h1.commands

            target_id = demo_h1._selected_id if demo_h1._selected_id is not None else 0
            points = scanner.data.ray_hits_w[target_id]
            weights_1 = attn_weights.permute(1,0,2,3)[..., 0, target_id].flatten()
            weights_2 = attn_weights.permute(1,0,2,3)[..., 1, target_id].flatten()
            marker_indices = (weights_1 * COLOR_SCALE).long()
            marker_indices = torch.clamp(marker_indices, min=0, max=9)
            heatmap_visualizer.visualize(
                translations=points,
                marker_indices=marker_indices
            )


if __name__ == "__main__":
    main()
    simulation_app.close()
