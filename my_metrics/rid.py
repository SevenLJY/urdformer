import sys, os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import numpy as np
from copy import deepcopy
from my_metrics.iou import sampling_iou
from my_metrics.giou import sampling_giou
from my_objects.dict_utils import (
    get_base_part_idx,
    get_bbox_vertices,
    find_part_mapping,
    compute_overall_bbox_size,
    zero_center_object,
    rescale_object
)

# [deprecated] old version
# def RID(requirement_dict, candidate_dict, num_states=1,
#         compare_handles=True, iou_include_base=False,
#         transform_use_plucker=False, rotation_fix_range=True,
#         sampling_scale_factor=10, num_samples=10000):
#     """
#     Compute the RID metric\n
#     This metric is the average sum of IoU between parts in the generated object and parts in the candidate object in the resting state\n

#     - requirement_dict: the requirement object dictionary\n
#     - candidate_dict: the candidate object dictionary\n
#     - num_states: the number of articulation states to compute the metric\n
#     - compare_handles (optional): whether to compare the handles\n
#     - iou_include_base (optional): whether to include the base part in the IoU computation\n
#     - transform_use_plucker (optional): whether to use plucker coordinate to transform the parts\n
#     - rotation_fix_range (optional): whether to fix the rotation range to 90 degrees for revolute joints\n
#     - sampling_scale_factor (optional): the object scale up factor for sampling\n
#         - This improves the sampling of the IoU for objects with small bounding boxes\n
#     - num_samples (optional): the number of samples to use\n

#     Return:\n
#     - score: the metric score, which is the overall average IoU over the parts and states\n
#         - The score is in the range of [0, 1], lower is better
#     """

#     # Make copies of the dictionaries to avoid modifying the original dictionaries
#     requirement_dict = deepcopy(requirement_dict)
#     candidate_dict = deepcopy(candidate_dict)

#     # Strip the handles from the requirement object if not comparing them
#     if not compare_handles:
#         requirement_dict = remove_handles(requirement_dict)

#     # Compute the scale factor by comparing the overall bbox size and scale the candidate object as a whole
#     requirement_bbox_size = compute_overall_bbox_size(requirement_dict)
#     candidate_bbox_size = compute_overall_bbox_size(candidate_dict)
#     scale_factor = requirement_bbox_size / candidate_bbox_size
#     rescale_object(candidate_dict, scale_factor)

#     # Scale up both objects for better point sampling
#     if sampling_scale_factor != 1:
#         rescale_object(requirement_dict, sampling_scale_factor)
#         rescale_object(candidate_dict, sampling_scale_factor)

#     # Record the indices of the base parts of the two objects
#     requirement_base_idx = get_base_part_idx(requirement_dict)
#     candidate_base_idx = get_base_part_idx(candidate_dict)

#     # Find mapping between the parts of the two objects based on closest bbox centers
#     # Force the base parts to be mapped to each other
#     part_mapping = find_part_mapping(requirement_dict, candidate_dict)
#     part_mapping[requirement_base_idx, :] = [candidate_base_idx, 0]

#     # Save the original bounding box vertices in rest pose
#     original_requirement_bbox_vertices = np.array([get_bbox_vertices(requirement_dict, i) for i in range(len(requirement_dict["diffuse_tree"]))])
#     original_candidate_bbox_vertices = np.array([get_bbox_vertices(candidate_dict, i) for i in range(len(candidate_dict["diffuse_tree"]))])

#     # Compute the sum of IoU between the generated object and the candidate object over a number of articulation states
#     num_parts_in_requirement = len(requirement_dict["diffuse_tree"])
#     iou_per_part_and_state = np.zeros((num_parts_in_requirement, num_states), dtype=np.float32)
#     # states = np.linspace(0, 1, num_states)


#     # Get a fresh copy of the bounding box vertices in rest pose
#     requirement_bbox_vertices = deepcopy(original_requirement_bbox_vertices)
#     candidate_bbox_vertices = deepcopy(original_candidate_bbox_vertices)

#     # Transform the objects to the current state using the joints
#     req_part_transfomrations = transform_all_parts(requirement_bbox_vertices, requirement_dict, 0, use_plucker=transform_use_plucker, rotation_fix_range=rotation_fix_range)
#     cand_part_transfomrations = transform_all_parts(candidate_bbox_vertices, candidate_dict, 0, rotation_fix_range=rotation_fix_range)

#     # Compute the IoU between the two objects using the transformed bounding boxes and the part mapping
#     for req_part_idx in range(num_parts_in_requirement):
#         # # Skip the base part
#         # if req_part_idx == requirement_base_idx:
#         #     continue

#         # Get the index of the corresponding part in the candidate object
#         cand_part_idx = int(part_mapping[req_part_idx, 0])

#         # Compute the sampling-based IoU between the two parts
#         # Always use a fresh copy of the bounding box vertices in rest pose in case dry_run=False is incorrectly set
#         req_part_bbox_vertices = deepcopy(original_requirement_bbox_vertices)[req_part_idx]
#         cand_part_bbox_vertices = deepcopy(original_candidate_bbox_vertices)[cand_part_idx]
#         iou = sampling_iou(req_part_bbox_vertices, cand_part_bbox_vertices, req_part_transfomrations[req_part_idx], cand_part_transfomrations[cand_part_idx], num_samples=num_samples)

#         iou_per_part_and_state[req_part_idx, 0] = iou

#     # Average the IoU over the states
#     per_part_iou_avg_over_states = np.sum(iou_per_part_and_state, axis=1) / num_states

#     # Average the IoU over the parts
#     if not iou_include_base:
#         per_part_iou_avg_over_states = np.delete(per_part_iou_avg_over_states, requirement_base_idx) # Remove the base part if specified

#     if len(per_part_iou_avg_over_states) == 0:
#         overall_avg_iou = 0
#     else:
#         overall_avg_iou = np.mean(per_part_iou_avg_over_states)

#     return 1. - overall_avg_iou


def compute_scores(dict1, dict2, iou_include_base=False, use_gIoU=False):
    '''Compute the RID and RIDD scores between two objects'''
    # Record the indices of the base parts of the two objects
    base_part_idx1 = get_base_part_idx(dict1)
    base_part_idx2 = get_base_part_idx(dict2)

    # Find mapping between the parts of the two objects based on closest bbox centers
    # Force the base parts to be mapped to each other
    part_mapping = find_part_mapping(dict1, dict2, use_hungarian=True)
    part_mapping[base_part_idx1, :] = [base_part_idx2, 0]

    # sum the minimum cost from hungarian matching
    min_cost = np.sum(part_mapping[:, 1])

    # Save the original bounding box vertices in rest pose
    original_bbox1_vertices = np.array(
        [get_bbox_vertices(dict1, i) for i in range(len(dict1["diffuse_tree"]))]
    )
    original_bbox2_vertices = np.array(
        [get_bbox_vertices(dict2, i) for i in range(len(dict2["diffuse_tree"]))]
    )

    # Compute the sum of IoU between the generated object and the candidate object over a number of articulation states
    num_parts_in_dict1 = len(dict1["diffuse_tree"])
    iou_per_part = np.zeros((num_parts_in_dict1,), dtype=np.float32)

    # Compute the IoU between the two objects using the transformed bounding boxes and the part mapping
    for part_idx1 in range(num_parts_in_dict1):
        # Get the index of the corresponding part in the candidate object
        part_idx2 = int(part_mapping[part_idx1, 0])

        # Compute the sampling-based IoU between the two parts
        # Always use a fresh copy of the bounding box vertices in rest pose in case dry_run=False is incorrectly set
        part_bbox1_vertices = deepcopy(original_bbox1_vertices)[part_idx1]
        part_bbox2_vertices = deepcopy(original_bbox2_vertices)[part_idx2]
        iou = (
            sampling_giou(part_bbox1_vertices, part_bbox2_vertices, [], [])
            if use_gIoU
            else sampling_iou(part_bbox1_vertices, part_bbox2_vertices, [], [])
        )

        iou_per_part[part_idx1] = iou

    # Average the IoU over the states
    per_part_iou_avg_over_states = iou_per_part

    # Average the IoU over the parts
    if not iou_include_base:
        per_part_iou_avg_over_states = np.delete(
            per_part_iou_avg_over_states, base_part_idx1
        )  # Remove the base part if specified

    if len(per_part_iou_avg_over_states) == 0:
        overall_avg_iou = 0
    else:
        overall_avg_iou = np.mean(per_part_iou_avg_over_states)

    return 1.0 - overall_avg_iou, min_cost


def RID(requirement_dict, candidate_dict, iou_include_base=False, use_gIoU=False):
    """
    Compute the normalized RID metric\n
    This metric is the average sum of IoU between parts in the generated object and parts in the candidate object in the resting state\n

    - requirement_dict: the requirement object dictionary\n
    - candidate_dict: the candidate object dictionary\n
    - iou_include_base (optional): whether to include the base part in the IoU computation, default is false\n

    Return:\n
    - RID: the RID score, which is the overall average IoU over the parts at resting state\n
        - The score is in the range of [0, 1], lower is better
    - RIDD: the RIDD score, the minimum distance between parts from hungarian matching\n
        - The score is the lower the better
    """

    # Make copies of the dictionaries to avoid modifying the original dictionaries
    requirement_dict = deepcopy(requirement_dict)
    candidate_dict = deepcopy(candidate_dict)

    # Zero center the objects
    zero_center_object(requirement_dict)
    zero_center_object(candidate_dict)

    # Compute the scale factor by comparing the overall bbox size and scale the candidate object as a whole
    requirement_bbox_size = compute_overall_bbox_size(requirement_dict)
    candidate_bbox_size = compute_overall_bbox_size(candidate_dict)
    scale_factor = requirement_bbox_size / candidate_bbox_size
    rescale_object(candidate_dict, scale_factor)

    return compute_scores(requirement_dict, candidate_dict, iou_include_base, use_gIoU)


if __name__ == "__main__":

    import json

    with open(
        "/localhome/jla861/Documents/projects/im-gen-ao/data/StorageFurniture/46787/train_v3.json"
    ) as f:
        gt = json.load(f)

    # with open('/localhome/jla861/Documents/projects/im-gen-ao/data/StorageFurniture/45633/train_v3.json') as f:
    #     gen = json.load(f)

    # rid, ridd = RID(gt, gen)
    # print(rid, ridd)

    res_json = (
        "exps/B3_part/l8h4_freg_soft1_l47_5e-5_5e-5_e100/images/test/epoch_099/35.json"
    )
    with open(res_json) as f:
        res = json.load(f)

    rids = []
    grids = []
    ridds = []
    for i, gen in enumerate(res["data"]):
        rid, ridd = RID(gt, gen)
        grid, _ = RID(gt, gen, use_gIoU=True)
        print(f"{i}: {rid}, {grid}, {ridd}")
        rids.append(rid)
        grids.append(grid)
        ridds.append(ridd)

    print(f"RID: {np.mean(rids)}, gRID: {np.mean(grids)}, RIDD: {np.mean(ridds)}")
