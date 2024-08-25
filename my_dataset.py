import copy
import json
import os
import time

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

PM2RGSEMREF = {"StorageFurniture": "cabinet_kitchen",
               "Oven": "oven",
               "Dishwasher": "dishwasher",
               "WashingMachine": "washer",
               "Refrigerator": "fridge",
               "Table": "table",
               "Microwave": "microwave",}

RGSEM2PMREF = {"cabinet_kitchen": "StorageFurniture",
               "oven": "Oven",
               "dishwasher": "Dishwasher",
               "washer": "WashingMachine",
               "fridge": "Refrigerator",
               "table": "Table",
               "microwave": "Microwave",}

RG2CODE = {"none": 0,
           "cabinet_kitchen": 1,
           "oven": 2,
           "dishwasher": 3,
           "washer": 4,
           "fridge": 5,
           "table": 6,
           "microwave": 7}

"""
base_names = [
        "none",
        "cabinet_kitchen",
        "oven",
        "dishwasher",
        "washer",
        "fridge",
        "oven_fan",
        "shelf_base",
    ]
"""


class PMDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.data_root = cfg.dataset.data_root
        self.split = split

        with open(cfg.dataset.data_split, "r") as f:
            data_split = json.load(f)
        with open(cfg.dataset.semantic_ref, "r") as f:
            self.semantic_ref = json.load(f)

        self.model_ids = data_split[split]
        self.index_map = []
        print(f"Loading {split} data...")
        start = time.time()
        self.data = self._load_data()
        print(f"Data loaded in {time.time() - start:.2f} seconds")

    def _load_data(self):
        data = {"anno": [], "img": [], "bbox": [], "supervision": [], "masks": [], "valid_parts_map": []}

        for idx, model_id in enumerate(tqdm(self.model_ids)):
            # print(f"Loading model {model_id}")
            with open(os.path.join(self.data_root, model_id, "train_v3.json"), "r") as f:
                model_data = json.load(f)

            # Make a version with base being the first part
            new_order, new_model_data = self._prepare_tree(copy.deepcopy(model_data))
            data["anno"].append(new_model_data)

            # Load the segmentation masks
            segs = np.load(os.path.join(self.data_root, model_id, "segs.npy"))

            for view_id in range(self.cfg.dataset.n_views):
                img_path = os.path.join(self.data_root, model_id, "imgs", f"{view_id:02d}.png")
                img = self._prepare_img(img_path)
                data["img"].append(img)

                # Get the bounding box for each part
                bbox_part_order_map, bboxes = self._get_bbox(segs[view_id], model_data["diffuse_tree"])
                data["valid_parts_map"].append(np.in1d(np.array(new_order), np.array(bbox_part_order_map))[1:])

                current_bboxes = np.zeros((self.cfg.dataset.num_max_parts, 4))
                if bboxes.shape[0] > 0:
                    current_bboxes[:bboxes.shape[0]] = bboxes
                data["bbox"].append(np.expand_dims(current_bboxes, 0))

                # Calculate masks (empty)
                padded_masks = np.zeros((self.cfg.dataset.num_max_parts, 14, 14))
                data["masks"].append(np.expand_dims(padded_masks, 0))

            self.index_map += [idx for _ in range(self.cfg.dataset.n_views)]

            mesh_types = self._get_mesh_types(new_model_data)
            padded_mesh_types = np.zeros((self.cfg.dataset.num_max_parts), dtype=np.int8)
            if len(mesh_types) > 0:
                padded_mesh_types[:len(mesh_types)] = mesh_types

            positions_min, positions_max = self._get_positions(new_model_data["diffuse_tree"], mesh_types)
            padded_positions_min = np.zeros((self.cfg.dataset.num_max_parts, 3), dtype=np.int8)
            padded_positions_max = np.zeros((self.cfg.dataset.num_max_parts, 3), dtype=np.int8)
            if len(positions_min) > 0:
                padded_positions_min[:len(positions_min)] = positions_min
                padded_positions_max[:len(positions_max)] = positions_max

            base_type = RG2CODE[PM2RGSEMREF[new_model_data["meta"]["obj_cat"]]]
            connectivity = self._get_connectivity(new_model_data["diffuse_tree"])
            padded_connectivity = np.zeros((self.cfg.dataset.num_max_parts + 1, self.cfg.dataset.num_max_parts + 1, self.cfg.URDFormer.num_relations), dtype=np.int8)
            if len(connectivity) > 0:
                padded_connectivity[:len(connectivity)] = connectivity

            supervision = {
                "positions": (padded_positions_min, padded_positions_max),
                "mesh_types": np.asarray(padded_mesh_types, dtype=np.int8),
                "base_type": np.asarray(base_type, dtype=np.int8),
                "connectivity": padded_connectivity
            }
            data["supervision"].append(supervision)

        return data

    def _reorder_tree(self, anno):
        # Make part with name "base" the first part
        # Modify children indices, parent index, parent ids
        # Do not modify the order of the parts, except moving other parts up to fill the gap
        tree = anno["diffuse_tree"]

        # First check if the base is already the first part
        if tree[0]["name"] == "base":
            return list(range(len(tree))), anno

        base_idx = -1
        for idx, part in enumerate(tree):
            if part["name"] == "base":
                base_idx = idx
                break
        base_part = tree[base_idx]
        base_part["id"] = 0
        base_part["parent"] = -1
        for i, child in enumerate(base_part["children"]):
            if child < base_idx:
                base_part["children"][i] += 1
        new_order = [base_idx]
        new_tree = [base_part]

        for idx, part in enumerate(tree):
            if idx == base_idx:
                continue
            new_order.append(idx)
            if part["id"] < base_idx:
                part["id"] += 1
            if part["parent"] < base_idx:
                part["parent"] += 1
            elif part["parent"] == base_idx:
                part["parent"] = 0
            # Update the children indices
            for i, child in enumerate(part["children"]):
                if child < base_idx:
                    part["children"][i] += 1
            new_tree.append(part)
        new_model_data = anno
        new_model_data["diffuse_tree"] = new_tree
        return new_order, new_model_data

    def _prepare_tree(self, model_data):
        # Makes part with name "base" the first part
        # Modifies children indices, parent index, parent ids
        # Removes parts annotated as wheel, shelf, tray
        # Does not modify the order of the parts, except moving other ids to fill the gap

        tree = model_data["diffuse_tree"]

        # First check if the base is already the first part
        # if tree[0]["name"] == "base":
        #    return list(range(len(tree))), tree

        base_idx = -1
        for idx, part in enumerate(tree):
            if part["name"] == "base":
                base_idx = idx
                break
        # Modify the base part itself
        base_part = tree[base_idx]
        base_part["id"] = 0
        base_part["parent"] = -1
        for i, child in enumerate(base_part["children"]):
            if child < base_idx:
                base_part["children"][i] += 1
        new_order = [base_idx]
        new_tree = [base_part]

        # Modify all parts in order, if required
        for idx, part in enumerate(tree):
            if idx == base_idx:
                continue
            new_order.append(idx)
            if part["id"] < base_idx:
                part["id"] += 1
            if part["parent"] < base_idx:
                part["parent"] += 1
            elif part["parent"] == base_idx:
                part["parent"] = 0
            # Update the children indices
            for i, child in enumerate(part["children"]):
                if child < base_idx:
                    part["children"][i] += 1
            new_tree.append(part)
        parts_to_remove = []
        for idx, part in enumerate(new_tree):
            if part["name"] in ["wheel", "shelf", "tray"]:
                parts_to_remove.append(idx)

        for idx in reversed(parts_to_remove):
            removed_part = new_tree.pop(idx)
            new_order.pop(idx)
            for part in new_tree:
                if part["id"] > idx:
                    part["id"] -= 1
                if part["parent"] == idx:
                    part["parent"] = removed_part["parent"]
                    new_tree[removed_part["parent"]]["children"].append(part["id"])
                elif part["parent"] > idx:
                    part["parent"] -= 1
                part["children"] = [child - 1 if child > idx else child for child in part["children"]]
                if idx in part["children"]:
                    part["children"].remove(idx)
        new_model_data = model_data
        new_model_data["diffuse_tree"] = new_tree
        return new_order, new_model_data

    def _get_mesh_types(self, anno):
        # 'none', 'drawer', 'doorL', 'doorR', 'handle', 'knob', 'washer_door', 'doorD', 'oven_door', 'doorU'
        obj_cat = anno["meta"]["obj_cat"]
        mesh_types = []
        for partInfo in anno["diffuse_tree"]:
            if partInfo["name"] == "base":
                continue
            if partInfo["name"] == "drawer":
                mesh_types.append(1)
            elif partInfo["name"] == "door":
                # Check the direction of opening
                axis_major_dir = np.argmax(np.abs(np.array(partInfo["joint"]["axis"]["direction"])))
                if axis_major_dir == 0 or axis_major_dir == 2:
                    # Axis is horizontal
                    # Now either doorD, or oven_door or doorU
                    if obj_cat == "Oven":
                        mesh_types.append(8)
                    else:
                        # Check whether axis origin is on top or bottom
                        if np.array(partInfo["joint"]["axis"]["origin"])[1] < np.array(partInfo["aabb"]["center"])[1]:
                            mesh_types.append(7)
                        else:
                            mesh_types.append(9)
                elif axis_major_dir == 1:
                    # Axis is vertical -> left/right motion
                    # Now check the origin relatively to part center x
                    if obj_cat == "WashingMachine":
                        mesh_types.append(6)
                    elif np.array(partInfo["joint"]["axis"]["origin"])[0] < np.array(partInfo["aabb"]["center"])[0]:
                        mesh_types.append(2)
                    else:
                        mesh_types.append(3)
            elif partInfo["name"] in ["handle", "knob"]:
                # As knob is not a separate label in our dataset, heuristic is impolemented
                aabb = np.array([np.array(partInfo["aabb"]["center"]) - np.array(partInfo["aabb"]["size"]) / 2,
                                np.array(partInfo["aabb"]["center"]) + np.array(partInfo["aabb"]["size"]) / 2])
                handle_diag_vec = aabb[1] - aabb[0]
                if np.abs(handle_diag_vec[0] - handle_diag_vec[1]) > 0.01 and partInfo["name"] == "handle":
                    mesh_types.append(4)
                else:
                    mesh_types.append(5)
        return mesh_types

    def _get_positions(self, tree, mesh_types):
        # Each part's position is voxelized relatively to it's parent, all in z-up coordinate system, 13*13
        # 0 is used for x all the time, as the depth of the object is not considered. So each position is (0, y, z)
        # Drawers, doorL, doorD, washer_door, oven_door are voxelized from bottom-left corner to top right corner
        # doorR is voxelized from bottom-right corner to top left corner
        # handle, knob only have center voxelized (so start and end are the same). Also order changes based on parent type
        # doorU is omitted in original checkpoint
        if len(mesh_types) == 0:
            return np.zeros([len(tree) - 1, 3], dtype=np.int8), np.zeros([len(tree) - 1, 3], dtype=np.int8)
        positions_min = np.zeros([len(tree) - 1, 3], dtype=np.int8)
        positions_max = np.zeros([len(tree) - 1, 3], dtype=np.int8)
        # Our data is y-up, we can easily voxelize it in y-up (using x and y as y and z in the output)
        for idx, part in enumerate(tree[1:]):
            mesh_type = mesh_types[idx]
            parent = tree[part["parent"]]
            parent_aabb = parent["aabb"]
            parent_bbox_min, parent_bbox_max = np.asarray(parent_aabb["center"]) - np.asarray(parent_aabb["size"]) / 2, np.asarray(parent_aabb["center"]) + np.asarray(parent_aabb["size"]) / 2
            parent_x_bins = np.linspace(parent_bbox_min[0], parent_bbox_max[0], 14)
            parent_y_bins = np.linspace(parent_bbox_min[1], parent_bbox_max[1], 14)

            part_aabb = part["aabb"]
            part_bbox_min, part_bbox_max = np.asarray(part_aabb["center"]) - np.asarray(part_aabb["size"]) / 2, np.asarray(part_aabb["center"]) + np.asarray(part_aabb["size"]) / 2

            if mesh_type == 3:
                # Voxelize part bbox min and max according to parent bins (right to left)
                right_bottom_corner = np.asarray([part_bbox_max[0], part_bbox_min[1]])
                left_top_corner = np.asarray([part_bbox_min[0], part_bbox_max[1]])
                # print(f"{idx}: {right_bottom_corner}, {left_top_corner}")

                right_bottom_vox_x = min(max(np.digitize(right_bottom_corner[0], parent_x_bins) - 1, 0), 12)
                right_bottom_vox_y = min(max(np.digitize(right_bottom_corner[1], parent_y_bins) - 1, 0), 12)

                left_top_vox_x = min(max(np.digitize(left_top_corner[0], parent_x_bins) - 1, 0), 12)
                left_top_vox_y = min(max(np.digitize(left_top_corner[1], parent_y_bins) - 1, 0), 12)

                # positions[idx] = np.asarray([[0, right_bottom_vox_x, right_bottom_vox_y], [0, left_top_vox_x, left_top_vox_y]])
                positions_min[idx] = np.asarray([0, right_bottom_vox_x, right_bottom_vox_y], dtype=np.int8)
                positions_max[idx] = np.asarray([0, left_top_vox_x, left_top_vox_y], dtype=np.int8)

            elif mesh_type in [4, 5]:
                part_center = np.asarray(part_aabb["center"])
                # print(f"{idx}: {part_center}")
                part_vox_center_x = min(max(np.digitize(part_center[0], parent_x_bins) - 1, 0), 12)
                part_vox_center_y = min(max(np.digitize(part_center[1], parent_y_bins) - 1, 0), 12)

                # positions[idx] = np.asarray([[0, part_vox_center_x, part_vox_center_y], [0, part_vox_center_x, part_vox_center_y]])
                if mesh_types[part["parent"] - 1] != 3:
                    positions_min[idx] = np.asarray([0, part_vox_center_x, part_vox_center_y], dtype=np.int8)
                    positions_max[idx] = np.asarray([0, part_vox_center_x, part_vox_center_y], dtype=np.int8)
                else:
                    # doorR - start from right to left
                    positions_min[idx] = np.asarray([0, 12 - part_vox_center_x, part_vox_center_y])
                    positions_max[idx] = np.asarray([0, 12 - part_vox_center_x, part_vox_center_y])
            else:
                # Voxelize part bbox min and max according to parent bins
                part_x_vox_min = min(max(np.digitize(part_bbox_min[0], parent_x_bins) - 1, 0), 12)
                part_y_vox_min = min(max(np.digitize(part_bbox_min[1], parent_y_bins) - 1, 0), 12)

                part_x_vox_max = min(max(np.digitize(part_bbox_max[0], parent_x_bins) - 1, 0), 12)
                part_y_vox_max = min(max(np.digitize(part_bbox_max[1], parent_y_bins) - 1, 0), 12)

                # positions[idx] = np.asarray([[0, part_x_vox_min, part_y_vox_min], [0, part_x_vox_max, part_y_vox_max]])
                positions_min[idx] = np.asarray([0, part_x_vox_min, part_y_vox_min], dtype=np.int8)
                positions_max[idx] = np.asarray([0, part_x_vox_max, part_y_vox_max], dtype=np.int8)
        return positions_min, positions_max

    def _get_connectivity(self, tree):
        # NOTE: assumption is that the base is always the first part, at index 0. Fulfilled by _reorder_tree
        connectivity_matrix = np.zeros([len(tree), self.cfg.dataset.num_max_parts + 1, self.cfg.URDFormer.num_relations], dtype=np.int8)
        for part_idx, part in enumerate(tree):
            if part["name"] == "base":
                # TODO: determine what to do in this case
                continue
            connectivity_matrix[part["id"]][part["parent"]][0] = 1
        return connectivity_matrix

    def _make_white_background(self, src_img):
        src_img.load()  # required for png.split()
        background = Image.new("RGB", src_img.size, (255, 255, 255))
        background.paste(src_img, mask=src_img.split()[3])  # 3 is the alpha channel
        return background

    def _prepare_img(self, img_path, tgt_size=256):
        '''
        Resize the image to 256x256, center crop to 224x224, and make the background white.
        '''
        img = Image.open(img_path)
        img = img.resize(size=(tgt_size, tgt_size), resample=Image.BICUBIC)
        img = img.crop(box=(16, 16, 240, 240))  # center crop to 224x224
        img = self._make_white_background(img)

        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(img)

        # Normalize
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        img_normalized = normalize(img_tensor)

        return img_normalized

    def _get_bbox(self, seg, tree):
        seg = cv2.resize(seg, (256, 256), interpolation=cv2.INTER_NEAREST)
        # center crop to 224x224
        center = seg.shape
        x = center[1] / 2 - 112
        y = center[0] / 2 - 112
        crop_seg = seg[int(y):int(y + 224), int(x):int(x + 224)]
        inst_id = 0
        bboxes = []
        part_order_map = []
        for part in tree:
            if part["name"] != "base":
                seg_id = self.semantic_ref["fwd"][part["name"]] * 100 + inst_id
                seg_part = crop_seg == seg_id
                # get the bounding box of the segmented region
                # Find the indices of the True values
                true_indices = np.argwhere(seg_part)
                if len(true_indices) == 0:
                    inst_id += 1
                    continue
                # Determine the bounding box coordinates
                min_row, min_col = true_indices.min(axis=0)
                max_row, max_col = true_indices.max(axis=0)
                # Create the bounding box coordinates
                bounding_box = np.asarray([min_row, min_col, max_row, max_col], dtype=np.float32)
                bboxes.append(bounding_box)
                part_order_map.append(inst_id)
            inst_id += 1
        normalized_bboxes = self._normalize_bbox(bboxes)
        bboxes = np.asarray(normalized_bboxes, dtype=np.float32)
        return part_order_map, bboxes

    def _normalize_bbox(self, bboxes, w=224, h=224):
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

    def __len__(self):
        return len(self.data["img"])

    def __getitem__(self, idx):
        input_data = {
            "img": self.data["img"][idx],
            "bbox": self.data["bbox"][idx],
            "masks": self.data["masks"][idx]
        }
        if np.all(self.data["valid_parts_map"][idx]):
            supervision = self.data["supervision"][self.index_map[idx]]
        else:
            # Create supervision copies, excluding the occluded parts
            supervision = self.data["supervision"][self.index_map[idx]].copy()
            valid_parts_map = self.data["valid_parts_map"][idx]
            mesh_types = np.zeros_like(supervision["mesh_types"])
            mesh_types[:np.sum(valid_parts_map)] = supervision["mesh_types"][np.where(valid_parts_map)[0]]
            supervision["mesh_types"] = mesh_types
            positions_min = np.zeros_like(supervision["positions"][0])
            positions_max = np.zeros_like(supervision["positions"][1])
            positions_min[:np.sum(valid_parts_map)] = supervision["positions"][0][np.where(valid_parts_map)[0]]
            positions_max[:np.sum(valid_parts_map)] = supervision["positions"][1][np.where(valid_parts_map)[0]]
            supervision["positions"] = (positions_min, positions_max)
            connectivity = supervision["connectivity"]
            connected_to_ids = np.where(connectivity[:, :, 0] == 1)[1]

            old_to_new_id = np.full(33, -1, dtype=int)
            valid_indices = np.where(valid_parts_map)[0]
            old_to_new_id[valid_indices + 1] = np.arange(len(valid_indices)) + 1
            old_to_new_id[0] = 0

            new_connectivity = np.zeros_like(connectivity)

            for i in range(1, 33):
                if i - 1 < len(valid_parts_map) and valid_parts_map[i - 1] and connected_to_ids[i - 1] >= 0:
                    new_id = old_to_new_id[i]
                    new_connected_to = old_to_new_id[connected_to_ids[i - 1]]
                    if new_connected_to >= 0:
                        new_connectivity[new_id, new_connected_to, 0] = 1

            supervision["connectivity"] = new_connectivity
        return input_data, supervision
