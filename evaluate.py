#!/usr/bin/env python3
# coding: utf-8

import argparse
import copy
import json
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np
import os
import sys
import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
import waymo_open_dataset

from collisionPossibility import getCollisionPossibility

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


## Usaage example:
## Basic flow is to 
## Step 1: Read through detection files to gather list of GT files to load
## Step 2: Load the required GT
## Step 3: Read detections again and calculate metrics
def plot_fig_paper(args):
    nets = ["fasterrcnn_resnet50_fpn", "yolov3", "maskrcnn_resnet50_fpn"]
    resolutions = ['320', '416', '608']
    r3_label = ["$R^3_1$", "$R^3_2$","$R^3_3$", "Recall" ]
    markers_list = ['o', 'x', '|', '.', 3]
    colors = ['red', 'blue', 'green', 'black']

    # First fetch list of gt files we need to load.
    gt_list = set()
    for ax_n, net in enumerate(nets):
        for ax_r, res in enumerate(resolutions):
            path = os.path.join(args.detections, net + '-gpu-' + res + "-det.json")

            # Step 1: Read through detection files to gather list of GT files to load
            _, fl = readDetections(path)
            gt_list.update(fl)

    fig, axes = plt.subplots(len(nets), len(resolutions), sharex=True, sharey=True, figsize=(12, 10))

    for rt_i, rt in enumerate(r3_thresh):
        # Skip over $R^3_1$, usually cause there are no obstacles that meet the criteria
        # if rt == 0.9:
            # continue
        print("Risk thresh = ", rt)

        # Step 2: Load the required GT
        groundtruth, dataset, waymo_frame_map = readWaymoGroundtruth(
                                                    gt_dir=args.groundtruth,
                                                    RRR=False,
                                                    acc_to_brake=args.max_acceleration,
                                                    t_reaction=args.computation_delay,
                                                    ts=args.timestep,
                                                    risk_thresh=float(rt),
                                                    gt_list=gt_list)
        for ax_n, net in enumerate(nets):
            for ax_r, res in enumerate(resolutions):
                path = os.path.join(args.detections, net + '-gpu-' + res + "-det.json")
                detections, file_list = readDetections(path)

                # Step 3: Read detections again and calculate metrics
                if rt == -1:
                    result = calcRecall(args, groundtruth, detections, False, True, dataset, waymo_frame_map, 0.8)
                else:
                    result = calcRecall(args, groundtruth, detections, True, True, dataset, waymo_frame_map, 0.8)

                net_name = net
                if '_' in net:
                    net_name = net.split('_')[0]
                title = net_name + ' ' + res
                axes[ax_n, ax_r].plot(conf_intervals, result, label=r3_label[rt_i], marker=markers_list[rt_i], color=colors[rt_i])
                axes[ax_n, ax_r].title.set_text(title)
                axes[ax_n, ax_r].set_xlim([0.5, 0.95])
                axes[ax_n, ax_r].set_xticks([0.55, 0.65, 0.75, 0.85], minor=True)
                axes[ax_n, ax_r].set_xticks([0.6, 0.7, 0.8, 0.9], minor=False)
                axes[ax_n, ax_r].tick_params(axis='y', left=True, right=True)
                axes[ax_n, ax_r].set_ylim([0, 1.1])
                axes[ax_n, ax_r].set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
                axes[ax_n, ax_r].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0], minor=False)
                if ax_r == 0: axes[ax_n, ax_r].set_ylabel('Recall')
                if ax_n == (len(nets) - 1): axes[ax_n, ax_r].set_xlabel('Confidence')


    # Some extra work to a unified legend
    r1 = mlines.Line2D([], [], color=colors[0], marker=markers_list[0], linestyle='None',
                          markersize=10, label=r3_label[0])
    r2 = mlines.Line2D([], [], color=colors[1], marker=markers_list[1], linestyle='None',
                          markersize=10, label=r3_label[1])
    r3 = mlines.Line2D([], [], color=colors[2], marker=markers_list[2], linestyle='None',
                          markersize=10, label=r3_label[2])
    r_def = mlines.Line2D([], [], color=colors[3], marker=markers_list[3], linestyle='None',
                          markersize=10, label=r3_label[3])
    fig.legend(handles=[r1, r2, r3, r_def], loc=7)
    fig.tight_layout()
    plt.savefig('rrr.svg', format="svg", bbox_inches="tight", dpi=300)


if __name__ == '__main__':
    args = get_args()
    plot_fig_paper(args)
