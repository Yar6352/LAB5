#! /usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan




class MyNode(Node):
    def __init__(self):
        super().__init__('bug_node')
        
        self.lidar_sub = self.create_subscription(
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

        self.cmd = None
    
    def lidar_callback(self, msg: LaserScan):
        data = np.array(msg.ranges)
        forward_angle = 120
        forward_dist = np.min(np.concatenate((data[:forward_angle//2], data[-forward_angle//2:])))
        cmd = Twist()
        if forward_dist > 0.3:
            cmd.linear.x = min(forward_dist - 0.25, 0.2)
        else:
            cmd.angular.z = 0.5
        self.cmd_pub.publish(cmd)
        

rclpy.init()

node = MyNode()

rclpy.spin(node)

