#! /usr/bin/env python3

import map_tools
import angle_tools
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt

from copy import deepcopy
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan




class MyNode(Node):
    def __init__(self):
        super().__init__('smart_bug_node')
        odom_sub = self.create_subscription(
            Odometry, 
            '/odom', 
            callback=self.odom_callback, 
            qos_profile=10,
        )

        map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            callback=self.map_callback,
            qos_profile=10
        )

        lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            callback=self.lidar_callback,
            qos_profile=10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            qos_profile=10
        )


        self.goal: np.ndarray = None
        self.last_traj: np.ndarray = None
        self.last_odom: Odometry = None
        self.last_lidar: LaserScan = None
        self.last_map: OccupancyGrid = None
        self.visit_map: np.ndarray = None
        self.map_global_origin: np.ndarray = None
        self.status = None # go / avoid / stop
        self.prev_status = None
        self.forward_angle = 100
        self.min_forward_dist = 0.5
        self.avoid_entry_angle = None
        self.avoid_entry_pos = None
    

    def lidar_callback(self, msg: LaserScan):
        self.last_lidar = msg
    
    def odom_callback(self, msg: Odometry):
        self.last_odom = msg

        if self.last_map is not None:
            map_tools.visit_point(
                grid=self.visit_map,
                global_point=np.array([self.last_odom.pose.pose.position.x, self.last_odom.pose.pose.position.y]),
                radius=0.5,
                map_global_origin=self.map_global_origin,
                r=self.last_map.info.resolution
                )
    
    def map_callback(self, msg: OccupancyGrid):
        self.last_map = msg

        if self.map_global_origin is None and self.last_odom is not None:
            robot_x = self.last_odom.pose.pose.position.x
            robot_y = self.last_odom.pose.pose.position.y
            map_x = self.last_map.info.origin.position.x
            map_y = self.last_map.info.origin.position.y
            self.map_global_origin = (map_x + robot_x, map_y + robot_y)
            self.visit_map = np.zeros((msg.info.height, msg.info.width), dtype=bool)

        if self.last_odom is not None and self.last_lidar is not None:
            self.run()
    
    def run(self):
        if self.status != self.prev_status:
            self.get_logger().info(f'current status is {self.status}')
            self.prev_status = self.status

        h, w = self.last_map.info.height, self.last_map.info.width
        r = self.last_map.info.resolution
        map = np.array(self.last_map.data, dtype=np.int16).reshape((h, w))
        robot_pos = np.array([self.last_odom.pose.pose.position.x, self.last_odom.pose.pose.position.y])
        
        if map.shape != self.visit_map.shape:
    
            dh = map.shape[0] - self.visit_map.shape[0]
            dw = map.shape[1] - self.visit_map.shape[1]

            if dh > 0:
                self.visit_map = np.concatenate((
                    self.visit_map,
                    np.zeros((dh, self.visit_map.shape[1]), dtype=bool)
                ), axis=0)
            elif dh < 0:
                self.visit_map = self.visit_map[:dh, :]
            
            if dw > 0:
                self.visit_map = np.concatenate((
                    self.visit_map,
                    np.zeros((self.visit_map.shape[0], dw), dtype=bool)
                ), axis=1)
            elif dw < 0:
                self.visit_map = self.visit_map[:, :dw]

        if self.goal is None:
            self.goal = map_tools.find_nearest_goal(
                map=map,
                visit_map=self.visit_map,
                robot_global_coordinates=robot_pos,
                min_dist=0.5,
                map_global_origin=self.map_global_origin,
                r=r,
                threshd= 30,
                threshu= 70,
            )

            if self.goal is None:
                self.status = 'stop'
            else:
                self.status = 'go'
                self.get_logger().info(f'current goal is {self.goal}')
        
        
        cmd = Twist()
        if self.status == 'go':
            
            angle_to_purp = angle_tools.angle_from_robot_to_purp(self.last_odom, self.goal)
            lidar_data = np.array(self.last_lidar.ranges)
            forward_dist = np.min(np.concatenate((lidar_data[:self.forward_angle//2], lidar_data[-self.forward_angle//2:])))
            if abs(angle_to_purp) < 7:
                cmd.linear.x = min(forward_dist - 0.25, 0.2)
            else:
                cmd.angular.z = 0.2 * (-1 if angle_to_purp < 0 else 1)
                self.cmd_pub.publish(cmd)
            
            p = np.array([
                    self.last_odom.pose.pose.position.x,
                    self.last_odom.pose.pose.position.y
            ])
            
            if np.linalg.norm(p - self.goal) < 0.5:
                self.goal = None
                self.status = 'stop'

            elif forward_dist < self.min_forward_dist:
                self.status = 'avoid'
                self.avoid_entry_pos = np.array([
                    self.last_odom.pose.pose.position.x,
                    self.last_odom.pose.pose.position.y
                    ])
                self.last_odom.pose.pose.orientation.z = 0.0
                self.last_odom.pose.pose.orientation.w = 0.0
                self.avoid_entry_angle = angle_tools.angle_from_robot_to_purp(
                    self.last_odom,
                    self.goal
                )
        
        elif self.status == 'avoid':
            lidar_data = np.array(self.last_lidar.ranges)
            forward_dist = np.min(np.concatenate((lidar_data[:self.forward_angle//2], lidar_data[-self.forward_angle//2:])))
            right_dist = np.min(lidar_data[-self.forward_angle//2 - 40:-self.forward_angle//2])

            if forward_dist < 0.4:
                cmd.angular.z = 0.2
            elif right_dist > 0.4:
                cmd.angular.z = -0.2
            else:
                cmd.linear.x = min(forward_dist - 0.25, 0.1)

            
            self.last_odom.pose.pose.orientation.z = 0.0
            self.last_odom.pose.pose.orientation.w = 0.0
            angle_to_purp = angle_tools.angle_from_robot_to_purp(
                self.last_odom,
                self.goal
            )
            p = np.array([
                    self.last_odom.pose.pose.position.x,
                    self.last_odom.pose.pose.position.y
            ])
            if np.linalg.norm(p - self.avoid_entry_pos) > 0.1 and abs(angle_to_purp - self.avoid_entry_angle) < 5:
                self.status = 'go'

        elif self.status == 'stop':
            self.get_logger().info('разведка окончена!')
        
        self.cmd_pub.publish(cmd)
          

rclpy.init()

node = MyNode()

rclpy.spin(node)
