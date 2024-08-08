'''
This file computes the IoU-based and centroid-distance-based metrics\n
'''
import sys, os
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import numpy as np
from copy import deepcopy
from my_objects.dict_utils import (
    get_base_part_idx,
    get_bbox_vertices,
    remove_handles,
    compute_overall_bbox_size,
    rescale_object,
    find_part_mapping,
    zero_center_object,
)
from my_objects.motions import transform_all_parts
from my_metrics.giou import sampling_giou, sampling_cDist


def IoU_cDist(
    gen_obj_dict,
    gt_obj_dict,
    num_states=5,
    compare_handles=False,
    iou_include_base=False,
    rotation_fix_range=True,
    num_samples=10000
):
    """
    Compute the IoU-based and centroid-distance-based metrics\n
    This metric is the average sum of IoU between parts in the two objects over the sampled articulation states and at the resting state\n

    - requirement_dict: the requirement object dictionary\n
    - candidate_dict: the candidate object dictionary\n
    - num_states: the number of articulation states to compute the metric\n
    - compare_handles (optional): whether to compare the handles\n
    - iou_include_base (optional): whether to include the base part in the IoU computation\n
    - rotation_fix_range (optional): whether to fix the rotation range to 90 degrees for revolute joints\n
    - sampling_scale_factor (optional): the object scale up factor for sampling\n
        - This improves the sampling of the IoU for objects with small bounding boxes\n
    - num_samples (optional): the number of samples to use\n
    - use_gIoU (optional): whether to use the generalized IoU instead of the standard IoU\n

    Return:\n
    - scores: a dictionary of the computed scores\n
        - "AID-IoU": the average IoU over the articulation states\n
        - "AID-cDist": the average centroid distance over the articulation states\n
        - "RID-IoU": the average IoU at the resting state\n
        - "RID-cDist": the average centroid distance at the resting state\n
    """

    # Make copies of the dictionaries to avoid modifying the original dictionaries
    requirement_dict = deepcopy(gen_obj_dict)
    candidate_dict = deepcopy(gt_obj_dict)

    # Strip the handles from the object if not comparing them
    if not compare_handles:
        requirement_dict = remove_handles(requirement_dict)
        candidate_dict = remove_handles(candidate_dict)

    # Zero center the objects
    zero_center_object(requirement_dict)
    zero_center_object(candidate_dict)

    # Compute the scale factor by comparing the overall bbox size and scale the candidate object as a whole
    requirement_bbox_size = compute_overall_bbox_size(requirement_dict)
    candidate_bbox_size = compute_overall_bbox_size(candidate_dict)
    scale_factor = requirement_bbox_size / candidate_bbox_size
    rescale_object(candidate_dict, scale_factor)

    # Record the indices of the base parts of the two objects
    requirement_base_idx = get_base_part_idx(requirement_dict)
    candidate_base_idx = get_base_part_idx(candidate_dict)

    # Find mapping between the parts of the two objects based on closest bbox centers
    # Force the base parts to be mapped to each other
    part_mapping = find_part_mapping(
        requirement_dict, candidate_dict, use_hungarian=True
    )
    part_mapping[requirement_base_idx, 0] = candidate_base_idx

    min_cost = np.sum(part_mapping[:, 1]) / len(part_mapping - 1)

    # Save the original bounding box vertices in rest pose
    original_requirement_bbox_vertices = np.array(
        [
            get_bbox_vertices(requirement_dict, i)
            for i in range(len(requirement_dict["diffuse_tree"]))
        ],
        dtype=np.float32,
    )
    original_candidate_bbox_vertices = np.array(
        [
            get_bbox_vertices(candidate_dict, i)
            for i in range(len(candidate_dict["diffuse_tree"]))
        ],
        dtype=np.float32,
    )

    # Compute the sum of IoU between the generated object and the candidate object over a number of articulation states
    num_parts_in_requirement = len(requirement_dict["diffuse_tree"])
    iou_per_part_and_state = np.zeros(
        (num_parts_in_requirement, num_states), dtype=np.float32
    )
    cDist_per_part_and_state = np.zeros(
        (num_parts_in_requirement, num_states), dtype=np.float32
    )
    states = np.linspace(0, 1, num_states)
    for state_idx, state in enumerate(states):

        # Get a fresh copy of the bounding box vertices in rest pose
        requirement_bbox_vertices = deepcopy(original_requirement_bbox_vertices)
        candidate_bbox_vertices = deepcopy(original_candidate_bbox_vertices)

        # Transform the objects to the current state using the joints
        req_part_transfomrations = transform_all_parts(
            requirement_bbox_vertices,
            requirement_dict,
            state,
            use_plucker=False,
            rotation_fix_range=rotation_fix_range,
        )
        cand_part_transfomrations = transform_all_parts(
            candidate_bbox_vertices,
            candidate_dict,
            state,
            rotation_fix_range=rotation_fix_range,
        )

        # Compute the IoU between the two objects using the transformed bounding boxes and the part mapping
        for req_part_idx in range(num_parts_in_requirement):

            # Get the index of the corresponding part in the candidate object
            cand_part_idx = int(part_mapping[req_part_idx, 0])

            # Always use a fresh copy of the bounding box vertices in rest pose in case dry_run=False is incorrectly set
            req_part_bbox_vertices = deepcopy(original_requirement_bbox_vertices)[
                req_part_idx
            ]
            cand_part_bbox_vertices = deepcopy(original_candidate_bbox_vertices)[
                cand_part_idx
            ]

            # Compute the sampling-based IoU between the two parts
            iou_per_part_and_state[req_part_idx, state_idx] = sampling_giou(
                    req_part_bbox_vertices,
                    cand_part_bbox_vertices,
                    req_part_transfomrations[req_part_idx],
                    cand_part_transfomrations[cand_part_idx],
                    num_samples=num_samples,
                )

            # Compute the centriod distance between the two matched parts
            cDist_per_part_and_state[req_part_idx, state_idx] = sampling_cDist(
                requirement_dict['diffuse_tree'][req_part_idx],
                candidate_dict['diffuse_tree'][cand_part_idx],
                req_part_transfomrations[req_part_idx],
                cand_part_transfomrations[cand_part_idx],
            )

    # IoU and cDist at the resting state
    per_part_iou_avg_at_rest = iou_per_part_and_state[:, 0]
    per_part_cDist_avg_at_rest = cDist_per_part_and_state[:, 0]


    # Average the IoU over the states
    per_part_iou_avg_over_states = np.sum(iou_per_part_and_state, axis=1) / num_states
    # Average the cDist over the states
    per_part_cDist_avg_over_states = (
        np.sum(cDist_per_part_and_state, axis=1) / num_states
    )

    # Remove the base part if specified
    if not iou_include_base:
        per_part_iou_avg_over_states = np.delete(
            per_part_iou_avg_over_states, requirement_base_idx
        )  
        per_part_iou_avg_at_rest = np.delete(
            per_part_iou_avg_at_rest, requirement_base_idx
        )
        per_part_cDist_avg_over_states = np.delete(
            per_part_cDist_avg_over_states, requirement_base_idx
        ) 
        per_part_cDist_avg_at_rest = np.delete(
            per_part_cDist_avg_at_rest, requirement_base_idx
        )

    aid_iou = float(np.mean(per_part_iou_avg_over_states)) if len(per_part_iou_avg_over_states) > 0 else 0
    aid_cdist = float(np.mean(per_part_cDist_avg_over_states)) if len(per_part_cDist_avg_over_states) > 0 else 1
    rid_iou = float(np.mean(per_part_iou_avg_at_rest)) if len(per_part_iou_avg_at_rest) > 0 else 0
    rid_cdist = float(np.mean(per_part_cDist_avg_at_rest)) if len(per_part_cDist_avg_at_rest) > 0 else 1

    return {
        "AID-IoU": aid_iou,
        "AID-cDist": aid_cdist,
        "RID-IoU": rid_iou,
        "RID-cDist": rid_cdist
    }


if __name__ == "__main__":

    import json

    with open(
        "/localhome/jla861/Documents/projects/im-gen-ao/data/Table/25144/train_v3.json"
    ) as f:
        gt = json.load(f)

    with open(
        "exps/B9/l8h4_l4_aug3d/images/test/epoch_099/15_25144/3/object.json"
        # "/localhome/jla861/Documents/projects/im-gen-ao/data/Table/25144/train_v3.json"
    ) as f:
        res = json.load(f)

    scores = IoU_cDist(gt, res, compare_handles=True, iou_include_base=False)
    print('AID-IoU', scores['AID-IoU'])
    print('AID-cDist', scores['AID-cDist'])
    print('RID-IoU', scores['RID-IoU'])
    print('RID-cDist', scores['RID-cDist'])
    print('min_cost', scores['min_cost'])
