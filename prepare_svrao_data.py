import os
import torch
import cv2 as cv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T
from segment_anything import sam_model_registry, SamPredictor

# load SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_ckpts/sam_vit_h_4b8939.pth")
sam.to('cuda')
predictor = SamPredictor(sam)
# load DINO model
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda()
dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True).cuda()
dinov2_T = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def load_resave_img(src_dir, f, dst_dir, resize=(224, 224)):
    img = cv.imread(os.path.join(src_dir, f))
    img = cv.resize(img, resize)
    cv.imwrite(os.path.join(dst_dir, 'input.png'), img)

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img_rgb

def _show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def _show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def sam_seg(img, save_dir, bbox_data, img_size=224):
    # prepare input image to SAM
    predictor.set_image(img)
    # load GT bbox
    norm_boxes = np.array(bbox_data['part_normalized_bbox'], dtype=np.float32)
    boxes = torch.tensor((norm_boxes * img_size).astype(int), device=predictor.device)
    # convert offset to the actual coordinates
    boxes[:, 2] += boxes[:, 0] 
    boxes[:, 3] += boxes[:, 1]
    # permute the dimensions
    part_boxes = boxes[:, [1, 0, 3, 2]]
    # prepare input bbox to SAM
    input_part_boxes = predictor.transform.apply_boxes_torch(part_boxes, img.shape[:2])
    # predict part masks given the part bboxes
    part_masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=input_part_boxes,
        multimask_output=False,
    )
    # convert to numpy
    np_part_masks = part_masks.cpu().numpy()
    np_part_boxes = part_boxes.cpu().numpy()
    # plot the result
    fig = plt.figure(figsize=(2.24,2.24), frameon=False, dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(img)
    for mask in np_part_masks:
        _show_mask(mask, ax, random_color=True)
    for box in np_part_boxes:
        _show_box(box, ax)
    fig.savefig(os.path.join(save_dir, 'segs.png'))
    # save the seg masks
    np.save(os.path.join(save_dir, 'segs.npy'), np_part_masks.squeeze(1))

def _extract_patch_features(img):
    # prepare input image to DINO
    img_input = dinov2_T(img).to('cuda').unsqueeze(0)
    with torch.no_grad():
        feat_reg = dinov2_vitb14_reg.forward_features(img_input)["x_norm_patchtokens"]
        feat = dinov2_vitb14.forward_features(img_input)["x_norm_patchtokens"]
    return feat_reg.cpu().numpy(), feat.cpu().numpy()

def dinov2_feat(img, save_dir):
    feat_reg, feat = _extract_patch_features(img)
    np.save(os.path.join(save_dir, 'dinov2_patch_reg.npy'), feat_reg)
    np.save(os.path.join(save_dir, 'dinov2_patch.npy'), feat)

if __name__ == '__main__':
    src_img_dir = 'images'
    save_root = '../data/Wild'
    os.makedirs(save_root, exist_ok=True)
    bbox_src = 'grounding_dino/labels_manual'
    for file in tqdm(os.listdir(src_img_dir)):
        if file.endswith('.png'):
            fname = 'urdformer_' + file[:-4]
            save_dir = os.path.join(save_root, fname)
            os.makedirs(save_dir, exist_ok=True)
            # load and resave image
            img = load_resave_img(src_img_dir, file, save_dir)
            # load bbox data
            bbox_data = np.load(os.path.join(bbox_src, file[:-4] + '.npy'), allow_pickle=True).item()
            # segment the part masks using SAM
            sam_seg(img, save_dir, bbox_data)
            # extract patch features using DINO
            dinov2_feat(img, save_dir)

        