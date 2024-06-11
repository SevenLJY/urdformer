import os
import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

root = '../data/URDFormer'
det = 'grounding_dino/labels_manual'
sam = sam_model_registry["vit_h"](checkpoint="sam_ckpts/sam_vit_h_4b8939.pth")
sam.to('cuda')
predictor = SamPredictor(sam)

for file in os.listdir('images'):
    if file.endswith('.png'):
        fname = file[:-4]
        os.makedirs(os.path.join(root, fname, 'imgs'), exist_ok=True)
        img = cv.imread(os.path.join('images', file))
        img = cv.resize(img, (224, 224))
        cv.imwrite(os.path.join(root, fname, 'imgs', '0.png'), img)
        label = np.load(os.path.join(det, fname + '.npy'), allow_pickle=True).item()
        norm_boxes = np.array(label['part_normalized_bbox'], dtype=np.float32)
        # # permute dimension: 0->2, 1->3, 2->0, 3->1
        # norm_boxes = norm_boxes[:, [1, 3, 0, 2]]
        boxes = torch.tensor((norm_boxes * 224).astype(int), device=predictor.device)
        
        # get segmentation
        img_input = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        obj_box_input = np.array([0,0,243,243])
        predictor.set_image(img_input)
        obj_masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=obj_box_input[None, :],
            multimask_output=False,
        )
        
        part_masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=boxes,
            multimask_output=False,
        )
        
        plt.figure(figsize=(10,10))
        plt.imshow(img_input)
        for mask in part_masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box in boxes:
            show_box(box.cpu().numpy(), plt.gca())
        plt.axis('off')
        plt.show()
        plt.imsave('test.png', img_input)