# Description: 
# input: mask data and json data
# output: png of half size of the mask data

import numpy as np
import cv2
import os
import json
# read mask npy

mask_path = "/home/hazel/Sources/Grounded-SAM-2/outputs/mask_data/"
json_path = "/home/hazel/Sources/Grounded-SAM-2/outputs/json_data/"

idx = 2
segmentation_objs = {0:"floor",1: "wall", 2: "chair"}

seg_obj = segmentation_objs[idx]

if seg_obj == "floor":
    obj_path = "/home/hazel/Sources/Grounded-SAM-2/outputs/floor_data_resize_by_2/"
elif seg_obj == "wall":
    obj_path = "/home/hazel/Sources/Grounded-SAM-2/outputs/wall_data_resize_by_2/"
elif seg_obj == "chair":
    obj_path = "/home/hazel/Sources/Grounded-SAM-2/outputs/chair_data_resize_by_2/"

os.makedirs(obj_path, exist_ok=True)

for i in range(0,200):

    mask_npy_path = os.path.join(mask_path, "mask_"+str(i)+".npy")     
    mask = np.load(mask_npy_path)
        
    # color map
    unique_ids = np.unique(mask)
            
    # get each mask from unique mask file
    all_object_masks = []
    
    for uid in unique_ids:
        if uid == 0: # skip background id
            continue
        else:
            object_mask = (mask == uid)
            all_object_masks.append(object_mask[None])
            
    if len(all_object_masks) == 0:
        # no object detected
        mask_resized = cv2.resize(mask.astype(np.uint8), (mask.shape[1]//2, mask.shape[0]//2), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(obj_path, str(i)+".png"), np.zeros_like(mask_resized))
        continue
    all_object_masks = np.concatenate(all_object_masks, axis=0)
        
    # load box information
    file_path = os.path.join(json_path, "mask_"+str(i)+".json")

    has_write = False

    with open(file_path, "r") as file:
        json_data = json.load(file)
        for obj_id, obj_item in json_data["labels"].items():
            if has_write:
                break
            class_name = obj_item["class_name"]
            mask_id = obj_item["instance_id"]
            if not mask_id in unique_ids:
                # in json but not in the mask
                continue
            if class_name == seg_obj:

                loc = (unique_ids == mask_id).nonzero()[0][0]
                obj_mask = all_object_masks[loc-1] # loc -1 because the first one is background
                h,w = obj_mask.shape
                obj_mask = cv2.resize(obj_mask.astype(np.uint8), (w//2, h//2), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(obj_path, str(i)+".png"), obj_mask*255)
                has_write = True
        if not has_write:
            mask_resized = cv2.resize(mask.astype(np.uint8), (mask.shape[1]//2, mask.shape[0]//2), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(obj_path, str(i)+".png"), np.zeros_like(mask_resized))
            

                    



