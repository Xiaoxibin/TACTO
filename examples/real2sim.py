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

log = logging.getLogger(__name__)

##a change
# width = np.float64(0.1)
# ## 将位置和方向数据转换为 float64
# position = np.array([0.5, 0.0, 0.2], dtype=np.float64)
# orientation = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
# Initialize a camera in the environment
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
    cv2.imwrite(f"{prefix}/vision/rgb_{step:04d}.png", rgb_img)
    np.save(f"{prefix}/vision/depth_{step:04d}.npy", depth_img)
    cv2.imwrite(f"{prefix}/tactile/color_{step:04d}.png", tactile_color)
    np.save(f"{prefix}/tactile/depth_{step:04d}.npy", tactile_depth)

##Control the gripper to gently grasp the object
def gentle_grasp(robot,width = 0.05):
    actions = robot.action_space.new()
    actions.end_effector.position = robot.get_states().end_effector.position
    actions.gripper_width = width  # 设置夹爪宽度为 0.05
    actions.gripper_force = 200
    robot.set_actions(actions)
    # Debugging: print out the gripper width before and after setting the action
    print(f"Desired gripper width: {width}")
    print(f"Gripper width after action: {robot.get_states().gripper_width}")
    p.stepSimulation()  # 运行仿真步，以确保动作应用
    time.sleep(0.1)
    print(f"Gripper width after action: {robot.get_states().gripper_width}")
    time.sleep(2)  # Allow time for the gripper to close
    # Check again after some time to see if it updated
    print(f"Gripper width after sleep: {robot.get_states().gripper_width}")

## Control the gripper to gently lift the object
def gentle_lift(robot, lift_height=0.1):
    # Use the go() method to set the end effector position and gripper width
    current_state = robot.get_states()
    current_position = current_state.end_effector.position
    width = current_state.gripper_width
    print(f"current_position: {current_position}")
    print(f"width: {width}")
    new_position = np.array([current_position[0], current_position[1], current_position[2] + lift_height])
    print(f"New position after lift height: {new_position}")
    # Use go() to move the end effector and set the gripper width
    robot.go(pos=new_position, ori=None, width=width, grip_force=100)

    time.sleep(2)  # Allow time for the object to be lifted

    # Debugging: print the new position and gripper width
    new_states = robot.get_states()
    print(f"New end effector position (after action): {new_states.end_effector.position}")
    print(f"New gripper width (after action): {new_states.gripper_width}")

######################################################################################################################### 主函数
# Load the config YAML file from examples/conf/grasp.yaml
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

    np.set_printoptions(suppress=True)

    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    robot.reset()

    panel = px.gui.RobotControlPanel(robot)
    panel.start()

    # Add camera
    view_matrix, projection_matrix = add_camera()

    # Create directories to save images
    os.makedirs("data/grasp/vision", exist_ok=True)
    os.makedirs("data/grasp/tactile", exist_ok=True)
    os.makedirs("data/lift/vision", exist_ok=True)
    os.makedirs("data/lift/tactile", exist_ok=True)

    # while True:
    ## Perform gentle grasp and capture images
    # gentle_grasp(robot, width = 0.05)
    # for step in range(8):
    #     rgb_img, depth_img = capture_image(view_matrix, projection_matrix)
    #     color_imgs, depth_imgs = digits.render()
    #     digits.updateGUI(color_imgs, depth_imgs)
    #     save_images(step, "data/grasp", rgb_img, depth_img, color_imgs[0], depth_imgs[0])
    #     time.sleep(0.1)

    ## Perform gentle lift and capture images
    gentle_lift(robot, lift_height=0.1)
    for step in range(8):
        rgb_img, depth_img = capture_image(view_matrix, projection_matrix)
        color_imgs, depth_imgs = digits.render()
        save_images(step, "data/lift", rgb_img, depth_img, color_imgs[0], depth_imgs[0])
        time.sleep(0.1)


if __name__ == "__main__":
    main()
