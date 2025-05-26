from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file', default='/root/pysot/experiments/siamrpn_r50_l234_dwxcorr_8gpu/config.yaml')
parser.add_argument('--snapshot', type=str, help='model name', default='/root/pysot/snapshot/checkpoint_e40.pth')
parser.add_argument('--dataset_dir', default='/root/pysot/training_dataset/satsot/train', type=str, help='dataset directory')
parser.add_argument('--output_dir', default='/root/pysot/output', type=str, help='directory to save result.')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = 'mps'

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu())['state_dict'])
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    # Parse init_rect

    for sub_dir in tqdm(os.listdir(args.dataset_dir)):
        if os.path.isdir(args.dataset_dir+'/'+sub_dir):
            gd = args.dataset_dir+'/'+sub_dir + '/groundtruth.txt'
            with open(gd, 'r') as f:
                gt = f.readlines()
            anchors = []
            for line in gt:
                if line.startswith('none'):
                    anchors.append([0, 0, 0, 0])
                else:
                    ls = list(map(int, line.split(',')))
                    anchors.append([ls[0], ls[1], ls[2], ls[3]])
            args.video_name = args.dataset_dir + '/' +sub_dir + '/img'
            init_rect = anchors[0]


        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir+'/'+sub_dir, exist_ok=True)

        first_frame = True
        frame_idx = 0

        for frame in get_frames(args.video_name):
            if first_frame:
                tracker.init(frame, init_rect)
                first_frame = False
            else:
                outputs = tracker.track(frame)
                if 'polygon' in outputs:
                    polygon = np.array(outputs['polygon']).astype(np.int32)
                    cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                True, (0, 255, 0), 3)
                    mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                    mask = mask.astype(np.uint8)
                    mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                    frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                else:
                    bbox = list(map(int, outputs['bbox']))
                    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                    if frame_idx == 0:
                        bboxes = []
                    bboxes.append({'frame': frame_idx, 'x': x, 'y': y, 'w': w, 'h': h})
                frame_idx += 1

        # Save bbox info to JSON using pandas
        df = pd.DataFrame(bboxes)
        video_name = os.path.basename(args.dataset_dir + '/' +sub_dir)
        json_path = os.path.join(args.output_dir, f'{video_name}.json')
        df.to_json(json_path, orient='records', force_ascii=False)

if __name__ == '__main__':
    main()