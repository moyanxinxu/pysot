import os
import json


all = {}
for sub_dir in os.scandir('/root/pysot/SatSOT'):
    if sub_dir.is_dir():
        gd = sub_dir.path + '/groundtruth.txt'

        with open(gd, 'r') as f:
            gt = f.readlines()

        anchors = []
        for line in gt:
            if line.startswith('none'):
                anchors.append([0, 0, 0, 0])
            else:
                ls = list(map(int, line.split(',')))
                anchors.append([ls[0], ls[1], ls[2] + ls[0], ls[3] + ls[1]])

        imgs = []
        for dir in os.scandir(sub_dir.path + '/img'):
            if dir.is_file():
                imgs.append(dir.name)

        answer = {}
        for img, anchor in zip(imgs, anchors):
            answer[img.split('.')[0]] = {"00000": anchor}

        # answer = {sub_dir.name: answer}
        all[sub_dir.name] = answer

with open('/root/pysot/SatSOT.json', 'w') as f:
    json.dump(all, f, indent=4)