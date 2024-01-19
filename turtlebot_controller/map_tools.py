#! /usr/bin/env python3

import cv2
import numpy as np

def global_to_map(map_global_origin, pos, r):
    x = pos[0] - map_global_origin[0]
    y = pos[1] - map_global_origin[1]
    return np.array([x / r, y / r])

def map_to_global(map_global_origin, pos, r):
    x = pos[0] + map_global_origin[0]
    y = pos[1] + map_global_origin[1]
    return np.array([x * r, y * r])



def find_nearest_goal(map:np.ndarray, visit_map: np.ndarray, robot_global_coordinates: np.ndarray, min_dist: float, map_global_origin, r, threshd: float, threshu:float):

    robot_map_coordinates = global_to_map(map_global_origin, robot_global_coordinates, r)
    res, d = None, None

    for y in range(map.shape[0]):
        for x in range(map.shape[1]):
            if not visit_map[y][x] and (threshd < map[y][x] < threshu):
                current_d = np.linalg.norm(robot_map_coordinates - np.array([x, y]))
                
                if res is None and ((min_dist / r) < current_d):
                    res = np.array([x, y])
                    d = current_d

                elif res is not None and ((min_dist / r) < current_d < d):
                    res = np.array([x, y])
                    d = current_d
                
    return map_to_global(map_global_origin, res, r)
    
                    
    

def visit_point(grid: np.ndarray, global_point: np.ndarray, radius: float, map_global_origin: np.ndarray, r: float) -> np.ndarray:
    map_point = global_to_map(map_global_origin, global_point, r)

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            p = np.array([x ,y])
            if np.linalg.norm(p - map_point) < radius:
                grid[y][x] = True

