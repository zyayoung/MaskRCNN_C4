from pycocotools.coco import COCO
import numpy as np
import cv2
from tqdm import tqdm

import os

dataDir='data/coco'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

cache_dir =  os.path.join(dataDir, dataType, "cache")
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
classes = ['__background__',  # always index 0
               'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
               'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
               'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
               'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
               'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
               'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
               'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
               'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
               'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
num_classes = len(classes)
# deal with class names
cats = [cat['name'] for cat in coco.loadCats(coco.getCatIds())]
class_to_coco_ind = dict(zip(cats, coco.getCatIds()))
class_to_ind = dict(zip(classes, range(num_classes)))
coco_ind_to_class_ind = dict([(class_to_coco_ind[cls], class_to_ind[cls])
                                for cls in classes[1:]])

print(coco_ind_to_class_ind)

image_ids = coco.getImgIds()
pbar = tqdm(total=len(image_ids))

def worker(index):
    im_info = coco.loadImgs(index)[0]
    annIds = coco.getAnnIds(imgIds=index, iscrowd=None)
    objs = coco.loadAnns(annIds)
    mask = np.zeros((im_info['height'], im_info['width']), np.uint32)
    id_db = np.zeros(100, np.uint8)
    for obj in objs:
        cat = coco_ind_to_class_ind[obj['category_id']]
        id_db[cat]+=1
        if id_db[cat] >= 1000:
            print("Warrning: id limit exceeded")
        ins_id = id_db[cat] + cat*1000
        # print(ins_id)
        mask[np.where(coco.annToMask(obj) != 0)] = ins_id
    np.savez_compressed(
        os.path.join(cache_dir, "{}".format(index)),
        mask,
    )
    pbar.update(8)

import multiprocessing as mp
pool = mp.Pool(processes=8)
pool.map(worker, image_ids)
