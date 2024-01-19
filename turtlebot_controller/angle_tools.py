#!/usr/bin/env python3.8

import numpy as np
from  nav_msgs.msg import Odometry
from math import (atan2, pi)

def get_purpose_angle(msg: Odometry, purpose_pos):
    '''
    calculating angle between x axis and vector from robot to purpose point
    '''
    
    pos = msg.pose.pose.position
    angle = atan2(purpose_pos[1] - pos.y, purpose_pos[0] - pos.x) * 180 / pi
    if angle < 0:
        angle += 360

    return angle

    
def get_robot_angle(msg: Odometry):
    '''
    calculating angle between x axis and robot line of sight
    '''

    quat = msg.pose.pose.orientation

    # that work because robot can rotate only around z axis
    angle = atan2(quat.z, quat.w) * 180 / pi * 2
    if angle < 0:
        angle = 360 - abs(angle)
    
    return angle

def create_vector(angle):
    return np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]) - np.array([0, 0])

def angle_from_robot_to_purp(msg: Odometry, purpose_pos):
    # anticlockwise is positive

    alhpaR = get_robot_angle(msg)
    alhpaP = get_purpose_angle(msg, purpose_pos)

    vecR = create_vector(alhpaR)
    vecP = create_vector(alhpaP)

    ans = np.rad2deg(np.arccos(np.dot(vecR, vecP)/np.linalg.norm(vecR)/np.linalg.norm(vecP)))

    clockwise = True
    if alhpaP - 0.1 <= (alhpaR + ans) % 360 <= alhpaP + 0.1:
        clockwise = False
    
    return - ans if clockwise else ans