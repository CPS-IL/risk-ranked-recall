#!/usr/bin/env python3
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import os

from rrr import r3_thresh, conf_intervals, get_args, readDetections, readWaymoGroundtruth, calcRecall


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
