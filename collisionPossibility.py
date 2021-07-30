#!/usr/bin/env python3
# coding: utf-8

import math
import numpy as np
from shapely.geometry import Polygon

def createTrajectory(x, v, timestep, iterations):
    trajectory = {}
    trajectory[0] = (x[0], x[1])
    for i in range(1, iterations):
        trajectory[timestep * i] = (trajectory[0][0] + (v[0] * timestep * i),
                                    trajectory[0][1] + (v[1] * timestep * i))
    return trajectory

def getDist(a, b):
    return math.sqrt(pow(a[0] - b[0], 2) +
                     pow(a[1] - b[1], 2))

def getIntersect3d(label, ego_center, obj_center):
    ego_2d = [[ 2 + ego_center[0],  1 + ego_center[1]],
              [ 2 + ego_center[0], -1 + ego_center[1]],
              [-2 + ego_center[0], -1 + ego_center[1]],
              [-2 + ego_center[0],  1 + ego_center[1]]]
    box = label.box
    cx = obj_center[0]
    cy = obj_center[1]
    cz = box.center_z
    l = box.length
    w = box.width
    h = box.height
    ry = box.heading
    x_corners = [l, l, -l, -l, l, l, -l, -l]
    y_corners = [w, -w, -w, w, w, -w, -w, w]
    z_corners = [h, h, h, h, -h, -h, -h, -h]
    R = np.array([[np.cos(ry), -np.sin(ry), 0], [np.sin(ry), np.cos(ry), 0], [0, 0, 1]])
    corners3d = np.vstack([x_corners, y_corners, z_corners]) / 2.0
    corners3d = (R @ corners3d).T + np.array([cx, cy, cz])
    p1 = Polygon([
        (corners3d[0][0], corners3d[0][1]),
        (corners3d[1][0], corners3d[1][1]),
        (corners3d[2][0], corners3d[2][1]),
        (corners3d[3][0], corners3d[3][1])])
    p2 = Polygon([
        (ego_2d[0][0], ego_2d[0][1]),
        (ego_2d[1][0], ego_2d[1][1]),
        (ego_2d[2][0], ego_2d[2][1]),
        (ego_2d[3][0], ego_2d[3][1])])
    return(p1.intersects(p2))

def getCollisionPossibility(ego_x, ego_v, obj_x, obj_v, label, a_max=7.5, t_compute=0.1, timestep=0.1, dist_limit=0):
    if dist_limit > 0:
        if getDist (ego_x, obj_x) > dist_limit:
            return "None"

    TTS = t_compute + math.sqrt(pow(ego_v[0], 2) + pow(ego_v[1], 2))/a_max
    iterations = int(TTS/timestep) + 1 # number of iterations
    ego_traj = createTrajectory(ego_x, ego_v, timestep, iterations)
    obj_traj = createTrajectory(obj_x, obj_v, timestep, iterations)
    dist_crit = math.sqrt(pow(label.box.length, 2) + pow(label.box.width, 2)) + 2.2 # m

    for i in range(iterations):
        if getIntersect3d(label, ego_traj[timestep * i], obj_traj[timestep * i]):
            return "Imminent"
        if getDist(ego_traj[timestep * i], obj_traj[timestep * i]) < \
                        (dist_crit + 2 * 0.5 * a_max * pow(timestep * i, 2)):
            return "Potential"
    return "None"
