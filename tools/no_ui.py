from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file', default='/root/pysot/experiments/siamrpn_r50_l234_dwxcorr_8gpu/config.yaml')
parser.add_argument('--snapshot', type=str, help='model name', default='/root/pysot/snapshot/checkpoint_e40.pth')
# parser.add_argument('--video_name', default='/root/pysot/demo/Mini61_2_vis.avi', type=str, help='videos or image files')
# parser.add_argument('--init_rect', type=str, help='initial rectangle (x,y,w,h)', default='315,266,25,8')
parser.add_argument('--video_name', default='/root/pysot/demo/bag.avi', type=str, help='videos or image files')
parser.add_argument('--init_rect', type=str, help='initial rectangle (x,y,w,h)', default='300,130,100,100')
parser.add_argument('--output_dir', default='/root/pysot/output', type=str, help='directory to save result frames')
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
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu())['state_dict'])
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    # Parse init_rect
    init_rect = list(map(int, args.init_rect.split(',')))

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

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
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)

            # Save the frame to the output directory
            output_path = os.path.join(args.output_dir, f'frame_{frame_idx:04d}.jpg')
            cv2.imwrite(output_path, frame)
            frame_idx += 1


if __name__ == '__main__':
    main()