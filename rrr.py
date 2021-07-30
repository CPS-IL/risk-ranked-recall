#!/usr/bin/env python3
# coding: utf-8

# Copyright © 2021, University of Illinois. All rights reserved.

import argparse
import copy
import json
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np
import os
from shapely.geometry import Polygon
import sys
import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
import waymo_open_dataset

tf.compat.v1.enable_eager_execution()
np.set_printoptions(threshold=sys.maxsize)

waymo_to_coco = {0: 10, 1: 1, 2: 0, 3: 8, 4: 5}  # from waymo to coco
conf_intervals = [.50, .55, .60, .65, .70, .75, .80, .85, .90, .95]

# Arbitrary thresholds for Risk Ranks, used for legacy reasons in internal plumbing
# [Imminent, Potential, None, No Risk Ranking]
r3_thresh = [0.9, 0.5, 0.1, -1]

def get_args():
    parser = argparse.ArgumentParser(
        description='Risk Ranked Recall for Waymo Open Dataset')
    parser.add_argument('--detections', '-det', required=True,
                        help='Path to detection file or directory containing output json')
    parser.add_argument('--groundtruth', '-gt', required=True,
                        help='Path to directory containing tfrecord files from the dataset.')
    parser.add_argument('--computation_delay', default=0.1,
                        help='Processing delay from sensor input to actuator output for the system.')
    parser.add_argument('--max_acceleration', default=7.5,
                        help='Max AV acceleration while braking')
    parser.add_argument('--timestep', default=0.1,
                        help='Discrete interval for risk analysis, lower values are compute heavy but better for accuracy of risk categorization.')
    parser.add_argument('--vcc', action='store_true',
                        help='Visualize corner cases frame by frame.')
    args = parser.parse_args()
    return args

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

def image_show(data, name, layout, cmap=None):
    plt.figure()
    """Show an image."""
    plt.imshow(tf.image.decode_jpeg(data), cmap=cmap)
    plt.title(name)
    plt.grid(False)
    plt.axis('off')

def visualize_waymo(frame, image_name, det, gt, metrics=None):
    # if not 0 in metrics[3]:
    #     return
    vel = 0.0
    for img in frame.images:
        vel += math.sqrt(pow(img.velocity.v_x,2) + pow(img.velocity.v_y,2) + pow(img.velocity.v_z,2))
    vel /= len(frame.images)

    for img in frame.images:
        if int(img.name) == int(image_name.split('_')[-1]):
            image = img

    ax = plt.subplot()

    # Iterate over the individual labels.
    for g in gt:
        # Draw the object bounding box.
        ax.add_patch(patches.Rectangle(
                    xy=(g.x1, g.y1),
                        width=abs(int(g.x2 - g.x1)),
                        height=abs(int(g.y2 - g.y1)),
                        linewidth=2,
                        edgecolor='green',
                        facecolor='none'))
        plt.text(g.x1, g.y1, str(g.rr))

    for d in det:
        # Draw the object bounding box.
        ax.add_patch(patches.Rectangle(
                    xy=(d.x1, d.y1),
                        width=abs(int(d.x2 - d.x1)),
                        height=abs(int(d.y2 - d.y1)),
                        linewidth=1,
                        edgecolor='red',
                        facecolor='none'))

    # Show the camera image.
    plt.imshow(tf.image.decode_jpeg(image.image))
    plt.title(open_dataset.CameraName.Name.Name(image.name))
    plt.grid(False)
    plt.axis('off')
    plt.savefig("dc_fn/" + image_name + ".png", bbox_inches="tight")
    # plt.show()


# Bounding Box 2D class
class Bb2d:
    def __init__(self, x1, y1, x2, y2, cat, conf=1, rr=1, id=None):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.cat = cat
        self.conf = conf
        self.rr = rr
        self.id = id

    def __str__(self):
        return str(self.x1) + ' '+str(self.y1) + ' '+str(self.x2) + ' '+str(self.y2) + ', cat:'+str(self.cat) + ', conf:'+str(self.conf)

    def plot(self, ax, c='black'):
        rect = patches.Rectangle((self.x1, self.y1), self.x2 - self.x1, self.y2 - self.y1,
            linewidth = 2, edgecolor = c, facecolor = 'none')
        ax.add_patch(rect)
# Intersection over union of 2 Bb2d

def bb_get_areas(boxA, boxB):
    xA = max(boxA.x1, boxB.x1)
    yA = max(boxA.y1, boxB.y1)
    xB = min(boxA.x2, boxB.x2)
    yB = min(boxA.y2, boxB.y2)
    interArea = max(0, xB - xA ) * max(0, yB - yA )
    boxAArea = (boxA.x2 - boxA.x1 ) * (boxA.y2 - boxA.y1 )
    boxBArea = (boxB.x2 - boxB.x1 ) * (boxB.y2 - boxB.y1 )
    return boxAArea, boxBArea, interArea

def bb_intersection_over_union(boxA, boxB):
    boxAArea, boxBArea, interArea = bb_get_areas(boxA, boxB)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Argument ordering (GT, Det)
def bb_intersection_over_groundtruth(boxA, boxB):
    boxAArea, boxBArea, interArea = bb_get_areas(boxA, boxB)
    iog = interArea / float(boxAArea)
    return iog

def waymo_label_to_BB(label, risk=-1):
    return Bb2d(float(label.box.center_x - 0.5 * label.box.length),
                float(label.box.center_y - 0.5 * label.box.width),
                float(label.box.center_x + 0.5 * label.box.length),
                float(label.box.center_y + 0.5 * label.box.width),
                int(waymo_to_coco[label.type]),float(1), risk, label.id)


# Given the groundtruth bbx and one predicted bb, it returns the maximum iou
# obtained and the index of the gt bbx
def find_best_match(target_bbs, bb, rrr=False):
    iou = []
    for tbb in target_bbs:
        if rrr:
            iou.append(bb_intersection_over_groundtruth(bb, tbb))
        else:
            iou.append(bb_intersection_over_union(bb, tbb))

    if iou == []:
        return 0, -1

    iou_max = max(iou)
    i = iou.index(iou_max)
    return iou_max, i

# Find best matching projected 3D label for 2D label.
def waymo_find_best_match_id(frame, camera_label, rrr=False, iou_thresh=0.5):
    for projected_labels in frame.projected_lidar_labels:
        if projected_labels.name == open_dataset.CameraName.FRONT:
            break
    target_bbs = []
    for label in projected_labels.labels:
        target_bbs.append(waymo_label_to_BB(label))
    bb = waymo_label_to_BB(camera_label)
    iou_max, i = find_best_match(target_bbs, bb, rrr)
    if iou_max >= iou_thresh:
        return target_bbs[i].id.split('_')[0]
    else:
        return None


def calcRecall(args, groundtruth, detections, rrr=False, classless=False, dataset=None, waymo_frame_map=None, iou_thresh=0.9):
    recall_res = []
    for i in range(len(conf_intervals)):
        recall_res.append([])

    for video in sorted(detections.keys()):
        if video in groundtruth:
            for image in sorted(detections[video].keys()):
                if image in groundtruth[video]:
                    gt_image = groundtruth[video][image]
                    for c_i, conf in enumerate(conf_intervals):
                        pred_bbs = []
                        for pred in detections[video][image]:
                            if pred.conf >= conf:
                                pred_bbs.append(pred)
                        for gt_bb in gt_image:
                            iou_max, iou_max_i = find_best_match(pred_bbs, gt_bb, rrr)
                            if iou_max >= iou_thresh:
                                recall_res[c_i].append(1)
                            else:
                                recall_res[c_i].append(0)
##########################################################################################
## This is an optional feature, to visualize corner cases.
                            if args.vcc and iou_max < iou_thresh:
                                visualize_waymo(waymo_frame_map[video][image], image,
                                    pred_bbs, groundtruth[video][image], recall_res)
                                # Disable this to keep seeing images,
                                # can be annoying for large inputs
                                sys.exit(1)
##########################################################################################                                
                else:
                    # print("Image not found in GT?", image)
                    pass
        else:
            print("Video not found in GT?", video)
            pass

    res = []
    for c_i, conf in enumerate(conf_intervals):
        if len(recall_res[c_i]) == 0:
            res.append(1)
        else:
            res.append(float(recall_res[c_i].count(1))/float(len(recall_res[c_i])))
    return res


def collisionCatGivenId(frame, id, acc_to_brake, t_reaction, ts, dist_limit=0):
    category = ""
    ego_x_x = 0
    ego_x_y = 0
    ego_v_x = 0
    ego_v_y = 0
    for img in frame.images:
        ego_v_x += img.velocity.v_x
        ego_v_y += img.velocity.v_y
    ego_v_x /= len(frame.images)
    ego_v_y /= len(frame.images)

    for label in frame.laser_labels:
        label_id = label.id.split('_')[0]
        if label_id == id:
            category = getCollisionPossibility((ego_x_x, ego_x_y),
                                               (ego_v_x, ego_v_y),
                                               (label.box.center_x, label.box.center_y),
                                               (label.metadata.speed_x, label.metadata.speed_y),
                                                label, acc_to_brake, t_reaction, ts, dist_limit)
            break
    if category == "Imminent": risk = 0.9
    elif category == "Potential": risk =  0.5
    else: risk = 0.1
    return risk

def readWaymoGroundtruth(gt_dir, RRR=False,
                         acc_to_brake=-7.5,
                         t_reaction=0.1, # 10 Hz
                         ts=0.1,
                         risk_thresh=-1,
                         gt_list = None):
    groundtruth = {}
    waymo_frame_map = {}    

    gtfiles = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(gt_dir)) for f in fn]
    tmp = []
    for f in gtfiles:
        if not os.path.isfile(f): continue
        if f.split('/')[-1] in gt_list:
            tmp.append(f)
    gtfiles = tmp

    for filename in gtfiles:
        assert('tfrecord' in filename)
        video_name = filename.split('/')[-1]
        groundtruth[video_name] = {}
        waymo_frame_map[video_name] = {}

        dataset = tf.data.TFRecordDataset(filename, compression_type='')
        for n_images, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            for camera_labels in frame.camera_labels:
                if camera_labels.name != open_dataset.CameraName.FRONT:
                    continue # Only use front images

                frame_name = str(frame.context.name) + '_' + \
                        str(n_images) + '_' + str(camera_labels.name)
                groundtruth[video_name][frame_name] = []
                waymo_frame_map[video_name][frame_name] = frame
                for label in camera_labels.labels:
                    id = waymo_find_best_match_id(frame, label)
                    if id == None:
                        continue
                    risk = collisionCatGivenId(frame, id, acc_to_brake, t_reaction, ts)
                    if risk == risk_thresh or risk_thresh == -1:
                        groundtruth[video_name][frame_name].append(
                                    waymo_label_to_BB(label, risk))
    count_obj = 0
    for v in groundtruth:
        for i in groundtruth[v]:
            count_obj += len(groundtruth[v][i])
    print("Objects in GT :", count_obj)
    return groundtruth, dataset, waymo_frame_map


def readDetections(infile):
    detections = {}
    file_list = set()

    if not os.path.isfile(infile):
        print("Detection file not present:", infile)
        sys.exit(-1)
    else:
        pass

    with open(infile, 'r') as f:
        y = json.load(f)
        for video in y:
            cur_video = {}
            video_name = video.split('/')[-1]
            file_list.add(video_name)
            for image in y[video]:
                if image.split('_')[-1] != "1": continue # Limit to front camera only.
                bbxs = []
                for det in y[video][image]:
                    # Format: x1, y1, x2, y2, cat, conf=1, rr=1, id=None
                    bb = Bb2d(det[0], det[1], det[2], det[3], det[6], det[5])
                    bbxs.append(bb)
                cur_video[image] = bbxs
            detections[video_name] = cur_video
    return detections, file_list
