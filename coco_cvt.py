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
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

image_ids = coco.getImgIds()
pbar = tqdm(total=len(image_ids))

def worker(index):
    im_info = coco.loadImgs(index)[0]
    annIds = coco.getAnnIds(imgIds=index, iscrowd=None)
    objs = coco.loadAnns(annIds)
    mask = np.zeros((im_info['height'], im_info['width']), np.uint32)
    id_db = np.zeros(100, np.uint8)
    for obj in objs:
        cat = obj['category_id']
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
    pbar.update()

import multiprocessing as mp
pool = mp.Pool(process=8)
pool.map(worker, index)
