import numpy as np
import PIL
from utils import detection_config
from PIL import Image
import argparse
from grounding_dino.detection import detector
from grounding_dino.post_processing import post_processing, visualize_bbox
import os, json
from tqdm import tqdm
import cv2 as cv

with open(
    "/localhome/jla861/Documents/projects/im-gen-ao/svr-ao/utils/semantic_ref.json", "r"
) as f:
    semantic_ref = json.load(f)["fwd"]


def get_part_bbox(model_id, img_id):
    root = "/localhome/jla861/Documents/projects/im-gen-ao/data"
    fpath = os.path.join(root, model_id, "segs.npy")
    segs = np.load(fpath)
    with open(os.path.join(root, model_id, "train_v3.json"), "r") as f:
        data = json.load(f)
        tree = data["diffuse_tree"]

    img_id = int(img_id)
    seg = segs[img_id]
    seg = cv.resize(seg, (256, 256), interpolation=cv.INTER_NEAREST)
    # center crop to 224x224
    center = seg.shape
    x = center[1] / 2 - 112
    y = center[0] / 2 - 112
    crop_seg = seg[int(y) : int(y + 224), int(x) : int(x + 224)]
    inst_id = 0
    bboxes = []
    for part in tree:
        if part["name"] != "base":
            seg_id = semantic_ref[part["name"]] * 100 + inst_id
            seg_part = crop_seg == seg_id
            ## get the bounding box of the segmented region
            # Find the indices of the True values
            true_indices = np.argwhere(seg_part)
            if len(true_indices) == 0:
                continue
            # Determine the bounding box coordinates
            min_row, min_col = true_indices.min(axis=0)
            max_row, max_col = true_indices.max(axis=0)
            # Create the bounding box coordinates
            bounding_box = [min_row, min_col, max_row, max_col]
            bboxes.append(bounding_box)
        inst_id += 1

    return bboxes

def normalize_bbox(bboxes, w=224, h=224):
    normalize_bboxes = []
    for bbox in bboxes:
        normalized = [
            bbox[0] / w,
            bbox[1] / h,
            (bbox[2] - bbox[0]) / w,
            (bbox[3] - bbox[1]) / h,
        ]
        normalize_bboxes.append(normalized)
    return normalize_bboxes

def prepare_gt_bbox(args):
    input_path = args.image_path
    manual_dir = "test_data/labels_gt"
    os.makedirs(manual_dir, exist_ok=True)
    for f in tqdm(os.listdir(input_path)):
        if f.endswith(".png"):
            data = {}
            src_img_path = os.path.join(input_path, f)
            dst_img_path = os.path.join(manual_dir, f[:-4])
            img = cv.imread(src_img_path)
            tokens = f[:-4].split("_")
            model_id = f"{tokens[0]}/{tokens[1]}"
            img_id = tokens[2]
            bboxes = get_part_bbox(model_id, img_id)
            visualize_bbox(img, dst_img_path, bboxes, thickness=2)
            normalize_bboxes = normalize_bbox(bboxes)
            data['bbox'] = bboxes
            data['part_normalized_bbox'] = normalize_bboxes
            np.save(f"{manual_dir}/{f[:-4]}.npy", data)


def evaluate(args, detection_args):
    input_path = args.image_path
    print(
        "************ Applying Finetuned (Model Soup) GroundingDINO *******************"
    )
    detector(args.scene_type, detection_args)
    # # # run postprocessing
    label_dir = args.pred_save_dir
    save_dir = "test_data/labels_filtered"
    os.makedirs(save_dir, exist_ok=True)
    post_processing(label_dir, input_path, save_dir)

def main():
    parser = argparse.ArgumentParser()
    # detection arguments
    parser.add_argument("--scene_type", default="all", type=str)
    parser.add_argument("--image_path", default="test_data/images", type=str)
    parser.add_argument("--pred_save_dir", default="test_data/labels_pred", type=str)

    ##################### IMPORTANT! ###############################
    # URDFormer replies on good bounding boxes of parts and ojects, you can achieve this by our annotation tool (~1min label per image)
    # We also provided our finetuned GroundingDINO (model soup version) to automate this. We finetuned GroundingDino on our generated dataset, and
    # apply model soup for the pretrained and finetuned GroundingDINO. However, the perfect bbox prediction is not gauranteed and will be our future work.

    args = parser.parse_args()
    detection_args = detection_config(
        args
    )  # leave the defult groundingDINO argument unchanged

    prepare_gt_bbox(args)
    # evaluate(args, detection_args)


if __name__ == "__main__":
    main()
