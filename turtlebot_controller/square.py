#! /usr/bin/env python3

import time
import numpy as np
import cv2
from matplotlib import pyplot as plt

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from std_srvs.srv import Empty




class MyNode(Node):
    def __init__(self):
        super().__init__('square_trajectory_node')
        map_sub = self.create_subscription(
            OccupancyGrid, 
            '/map', 
            callback=self.map_callback, 
            qos_profile=10,
            )
        odom_sub = self.create_subscription(
            Odometry, 
            '/odom', 
            callback=self.odom_callback, 
            qos_profile=10,
            )
        self.last_map: OccupancyGrid = None
        self.last_odom: Odometry = None
        self.map_global_origin: tuple = None

    def _draw_map(self, show=False):
        if self.last_map is None or self.last_odom is None:
            return
        if self.map_global_origin is None:
            robot_x = self.last_odom.pose.pose.position.x
            robot_y = self.last_odom.pose.pose.position.y
            map_x = self.last_map.info.origin.position.x
            map_y = self.last_map.info.origin.position.y
            self.map_global_origin = (map_x + robot_x, map_y + robot_y)
        
        h, w = self.last_map.info.height, self.last_map.info.width
        r = self.last_map.info.resolution

        arr = np.array(self.last_map.data, dtype=np.int16).reshape((h, w))
        arr[arr < 0] = 200

        dx = self.last_odom.pose.pose.position.x - self.map_global_origin[0]
        dy = self.last_odom.pose.pose.position.y - self.map_global_origin[1]

        show = True
        if show:
            plt.clf()
            plt.imshow(arr)
            plt.scatter(
                x=dx/r,
                y=dy/r, 
                s=50, 
                c='red',
                marker='*')
            plt.pause(0.01)
            plt.show(block=False)
    
    def map_callback(self, msg: OccupancyGrid):
        self.last_map = msg
        self._draw_map()
    
    def odom_callback(self, msg: Odometry):
        self.last_odom = msg
        self._draw_map()

rclpy.init()

node = MyNode()

rclpy.spin(node)

