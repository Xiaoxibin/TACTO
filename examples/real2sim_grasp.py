# -*- coding: utf-8 -*-

import time
import logging

import hydra
import numpy as np
import pybullet as p

import pybulletX as px
import tacto

from sawyer_gripper import SawyerGripper

import cv2
import os
os.chdir('/home/xxb/tacto-main/examples')

## 模型相关包
import torch
import torchvision.transforms as transforms
from PIL import Image
import pyrealsense2 as rs
import sys
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import json
from PIL import Image
import matplotlib.pyplot as plt
import yaml

sys.dont_write_bytecode = True
sys.path.append( os.path.abspath(os.path.join(os.path.dirname(__file__),"../")) )
from utils import model_factory
from utils.test_data_loader import GraspingSlidingDataset

log = logging.getLogger(__name__)
### 检查 CUDA 是否可用
cuda_avail = torch.cuda.is_available()

def move_to_position(robot, digits, end_pos, steps=50, width=None, grip_force=20, view_matrix=None, projection_matrix=None):
    """Moves the end effector to end_pos in a specified number of steps."""
    current_pos = robot.get_states().end_effector.position
    delta = (end_pos - current_pos) / steps

    for i in range(steps):
        current_pos += delta
        robot.go(pos=current_pos, ori=None, width=width, grip_force=grip_force)

        ## 强制更新仿真状态
        p.stepSimulation()

        ## Continuously update and display tactile images during the motion
        color, depth = digits.render()
        digits.updateGUI(color, depth)
        time.sleep(0.05)  ## Small delay between steps for smooth motion

        ## 捕捉和保存图像
        if view_matrix is not None and projection_matrix is not None:
            rgb_img, depth_img = capture_image(view_matrix, projection_matrix)
            save_images(i, "data/move", rgb_img, depth_img, color[0], depth[0])

def add_camera():
    # Camera parameters
    cam_target_pos = [0.5, 0, 0.02]  # Target position
    cam_distance = 0.2         # Distance from target position
    cam_yaw = 90                # Yaw angle  'z'
    cam_pitch = -30             # Pitch angle 'y'
    cam_roll = 0                # Roll angle  'x'
    cam_fov = 60                # Field of view
    cam_aspect = 1.0            # Aspect ratio
    cam_near_plane = 0.01       # Near clipping plane
    cam_far_plane = 100         # Far clipping plane

    # Calculate camera view matrix
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target_pos,
        distance=cam_distance,
        yaw=cam_yaw,
        pitch=cam_pitch,
        roll=cam_roll,
        upAxisIndex=2
    )

    # Calculate projection matrix
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=cam_fov,
        aspect=cam_aspect,
        nearVal=cam_near_plane,
        farVal=cam_far_plane
    )

    return view_matrix, projection_matrix

# Capture camera image
def capture_image(view_matrix, projection_matrix):
    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
        width=640,
        height=480,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix
    )
    return rgb_img, depth_img

# Save images
def save_images(step, prefix, rgb_img, depth_img, tactile_color, tactile_depth):
    cv2.imwrite(f"{prefix}/rgb/rgb_{step:04d}.png", rgb_img)
    # np.save(f"{prefix}/vision/depth_{step:04d}.npy", depth_img)
    cv2.imwrite(f"{prefix}/tactile/color_{step:04d}.png", tactile_color)
    # np.save(f"{prefix}/tactile/depth_{step:04d}.npy", tactile_depth)

##################################################################################### 调整末端执行器宽度
def adjust_end_effector_width(predicted_class, width):
    if predicted_class == 1:
        return width
    elif predicted_class == 0:
        return width + 0.03
    elif predicted_class == 2:
        return width - 0.03
    # else:
    #     return [0, 0, 0, 0, 0, 0, 0, 0]  # 默认返回一个全零的配置

#################################################################################### 执行捏举动作并保存图像
def execute_grasp_and_lift(robot, digits, view_matrix, projection_matrix, steps, width=0.04, grip_force=100):
    """执行捏和举的操作，并保存图像"""
    # Step 2: Close the gripper to grasp the object
    move_to_position(robot, digits, steps['grasp'], width=width, grip_force=grip_force)
    time.sleep(0.1)  # Wait for the gripper to close
    for step in range(8):
        rgb_img, depth_img = capture_image(view_matrix, projection_matrix)
        color_imgs, depth_imgs = digits.render()
        save_images(step, "data/grasping", rgb_img, depth_img, color_imgs[0], depth_imgs[0])
        time.sleep(0.1)

    # Step 3: Lift the object by 2 cm
    move_to_position(robot, digits, steps['lift'], width=width, grip_force=grip_force)
    for step in range(8):
        rgb_img, depth_img = capture_image(view_matrix, projection_matrix)
        color_imgs, depth_imgs = digits.render()
        save_images(step, "data/sliding", rgb_img, depth_img, color_imgs[0], depth_imgs[0])
        time.sleep(0.1)

##################################################################################### 滑移检测模型预测函数
def test_module(params, model_path, data_path, width):

    # 图片处理
    transform_rgb = transforms.Compose([
        transforms.Resize((120, 160)),
        transforms.ToTensor()
    ])

    transform_tactile = transforms.Compose([
        transforms.Resize((150, 200)),
        transforms.ToTensor()
    ])


    # 检查是否使用 GPU
    if params['use_gpu'] == 1 and cuda_avail:
        device = torch.device("cuda:0")
        use_gpu = True
    else:
        device = torch.device("cpu")
        use_gpu = False

    # 加载模型
    if params['Modality'] == "Combined":
        NN_model, model_params = model_factory.get_model(params, use_gpu)
    
    # 加载预训练模型权重
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    NN_model.load_state_dict(state_dict['model'], strict=False)
    
    if use_gpu:
        NN_model = NN_model.cuda()

    NN_model.eval()
    
    # 预处理输入图像
    # data_path = '/home/xxb/Downloads/data'
    dataset = GraspingSlidingDataset(data_path, transform_rgb=transform_rgb, transform_tactile=transform_tactile)

    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    data = next(iter(dataloader))

    # 运行模型推理
    with torch.no_grad():
        if params['Modality'] == "Combined":
            output = NN_model(data[0], data[1], data[2], data[3], torch.tensor([[width * 100]]))
        _, predicted = torch.max(output.data, 1)
    
    # 输出预测结果
    print(f"Output: {output}")
    print(f"Predicted: {predicted}")

    # 提取 output 中的最大值索引（类别）
    output_class = torch.argmax(output, dim=1).item()

    # 提取 predicted 中的值
    predicted_class = predicted.item()

    # 打印结果
    print(f"Output class: {output_class}")
    print(f"Predicted class: {predicted_class}")
    return output_class, predicted_class

@hydra.main(config_path="conf", config_name="grasp")
def main(cfg):
    # Initialize digits
    digits = tacto.Sensor(**cfg.tacto)

    # Initialize World
    log.info("Initializing world")
    px.init()

    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    robot = SawyerGripper(**cfg.sawyer_gripper)

    # [21, 24]
    digits.add_camera(robot.id, robot.digit_links)

    # Add object to pybullet and digit simulator
    obj = px.Body(**cfg.object)
    digits.add_body(obj)

    ## Add camera
    view_matrix, projection_matrix = add_camera()
    np.set_printoptions(suppress=True)

    # Start the simulation thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    # Reset the robot to its initial state
    robot.reset()
 
    ## Initial tactile image rendering
    color, depth = digits.render()
    digits.updateGUI(color, depth)
    time.sleep(1)

    ## Initial rgb image rendering
    rgb_img, depth_img = capture_image(view_matrix, projection_matrix)
    color_imgs, depth_imgs = digits.render()

    # Create directories to save images
    os.makedirs("data/grasping/rgb", exist_ok=True)
    os.makedirs("data/grasping/tactile", exist_ok=True)
    os.makedirs("data/sliding/rgb", exist_ok=True)
    os.makedirs("data/sliding/tactile", exist_ok=True)

    ## 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    ## 32 张图片路径
    save_dir = os.path.join(current_dir, 'data')

    ## VIVIT 模型相关路径
    ## 加载配置文件
    with open('/home/xxb/DeformableObjectsGrasping-master/src/grasping_framework/config_cluster.yaml', 'r') as file:
        params = yaml.safe_load(file)
    ## 模型路径
    model_path = '/home/xxb/DeformableObjectsGrasping-master/src/grasping_framework/Trained_Model/vivit_fdp_two/03_07_2024__09_54_46/vivit_fdp_two46.pt'
    ## 输入图像路径
    data_path = save_dir

    ## 末端执行器夹爪宽度
    width_tomato = 0.04

    ## 获取机器人当前状态
    current_state = robot.get_states()
    end_effector_position = current_state.end_effector.position


    ## 预定义步骤位置
    steps = {
        'grasp': np.array([end_effector_position[0], end_effector_position[1], end_effector_position[2] - 0.05]),
        'lift': np.array([end_effector_position[0], end_effector_position[1], end_effector_position[2] - 0.03])
    }

    move_to_position(robot, digits, steps['grasp'], width=0.1, grip_force=100)

    while True:
        ## 执行捏和举的操作
        execute_grasp_and_lift(robot, digits, view_matrix, projection_matrix, steps, width=width_tomato, grip_force=100)

        ## VIVIT模型预测
        output_class, predicted_class = test_module(params, model_path, data_path, width = width_tomato)

        ## 检查预测结果，并调整夹爪宽度或退出循环
        if predicted_class == 1:
            print("成功抓取物体，退出循环")
            break
        else:
            print(f"滑移检测到, 预测类别: {predicted_class}, 调整夹爪宽度并重试")
            adjust_width = adjust_end_effector_width(predicted_class, width=0.04)
            move_to_position(robot, digits, steps['grasp'], width=adjust_width, grip_force=100)

    ## Keep the simulation running to observe the result
    while True:
        color, depth = digits.render()
        digits.updateGUI(color, depth)
        time.sleep(0.1)

if __name__ == "__main__":
    main()
