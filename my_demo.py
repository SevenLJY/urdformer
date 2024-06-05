# import numpy as np

# label3 = np.load("grounding_dino/labels_manual/label3.npy", allow_pickle=True).item()
# nbbox = label3["part_normalized_bbox"]
# bbox = label3["bbox"]


import numpy as np
import PIL
from urdformer import URDFormer
import json
import torch
import cv2
import pybullet as p
from utils import visualization_global, visualization_parts, detection_config, create_obj
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
import argparse
from texture import load_texture
from utils import write_numpy
from utils import write_urdfs
from grounding_dino.detection import detector
from grounding_dino.post_processing import post_processing, summary_kitchen
import os
import time
import glob

# integrate the extracted texture map into URDFormer prediction
def evaluate_real_image(image_tensor, bbox, masks, tgt_padding_mask, tgt_padding_relation_mask, urdformer, device):
    rgb_input = image_tensor.float().to(device).unsqueeze(0)
    bbox_input = torch.tensor(bbox).float().to(device).unsqueeze(0)
    masks_input = torch.tensor(masks).float().to(device).unsqueeze(0)

    tgt_padding_mask = torch.logical_not(tgt_padding_mask)
    tgt_padding_mask = torch.tensor(tgt_padding_mask).to(device).unsqueeze(0)


    tgt_padding_relation_mask = torch.logical_not(tgt_padding_relation_mask)
    tgt_padding_relation_mask = torch.tensor(tgt_padding_relation_mask).to(device).unsqueeze(0)

    position_x_pred, position_y_pred, position_z_pred, position_x_end_pred, position_y_end_pred, position_z_end_pred, mesh_pred, parent_cls, base_pred = urdformer(rgb_input, bbox_input, masks_input, 2)
    position_pred_x = position_x_pred[tgt_padding_mask].argmax(dim=1)
    position_pred_y = position_y_pred[tgt_padding_mask].argmax(dim=1)
    position_pred_z = position_z_pred[tgt_padding_mask].argmax(dim=1)

    position_pred_x_end = position_x_end_pred[tgt_padding_mask].argmax(dim=1)
    position_pred_y_end = position_y_end_pred[tgt_padding_mask].argmax(dim=1)
    position_pred_z_end = position_z_end_pred[tgt_padding_mask].argmax(dim=1)

    mesh_pred = mesh_pred[tgt_padding_mask].argmax(dim=1)

    base_pred = base_pred.argmax(dim=1)

    parent_pred = parent_cls[tgt_padding_relation_mask]

    position_pred = torch.stack([position_pred_x, position_pred_y, position_pred_z]).T
    position_pred_end = torch.stack([position_pred_x_end, position_pred_y_end, position_pred_z_end]).T

    return position_pred.detach().cpu().numpy(), position_pred_end.detach().cpu().numpy(), mesh_pred.detach().cpu().numpy(), parent_pred.detach().cpu().numpy(), base_pred.detach().cpu().numpy()

def image_transform():
    """Constructs the image preprocessing transform object.

    Arguments:
        image_size (int): Size of the result image
    """
    # ImageNet normalization statistics
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocessing = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

    return preprocessing

def evaluate_parts_with_masks(data_path, cropped_image):
    max_bbox = 32
    num_roots = 1
    data = np.load(data_path, allow_pickle=True).item()
    img_pil = PIL.Image.fromarray(cropped_image).resize((224, 224))

    img_transform = image_transform()
    image_tensor = img_transform(img_pil)

    bbox = []
    resized_mask = []
    for boxid, each_bbox in enumerate(data['part_normalized_bbox']):
        bbox.append(each_bbox)
        resized = np.zeros((14, 14))
        resized_mask.append(resized)
    padded_bbox = np.zeros((max_bbox, 4))
    padded_bbox[:len(bbox)] = bbox

    padded_masks = np.zeros((max_bbox, 14, 14))
    padded_masks[:len(resized_mask)] = resized_mask

    tgt_padding_mask = torch.ones([max_bbox])
    tgt_padding_mask[:len(bbox)] = 0.0
    tgt_padding_mask = tgt_padding_mask.bool()

    tgt_padding_relation_mask = torch.ones([max_bbox + num_roots])
    tgt_padding_relation_mask[:len(bbox) + num_roots] = 0.0
    tgt_padding_relation_mask = tgt_padding_relation_mask.bool()

    return image_tensor, np.array([padded_bbox]), np.array([padded_masks]), tgt_padding_mask, tgt_padding_relation_mask

def get_binary_relation(global_relations, position_pred_global, num_roots):

    new_relations = np.zeros((len(position_pred_global) + num_roots, len(position_pred_global) + num_roots, 6))
    for obj_id, position in enumerate(position_pred_global):
        each_parent = np.unravel_index(np.argmax(global_relations[num_roots + obj_id]),
                                       global_relations[num_roots + obj_id].shape)
        parent_id = each_parent[0]
        relation_id = each_parent[1]
        new_relations[obj_id + num_roots, parent_id, relation_id] = 1
    return new_relations

def get_binary_relation_parts(part_relations, position_pred_part, num_roots):
    all_new_relations = []
    for obj_id, each_position in enumerate(position_pred_part):
        new_relations = np.zeros((len(position_pred_part[obj_id]) + num_roots, len(position_pred_part[obj_id]) + num_roots, 6))
        for part_id, position in enumerate(position_pred_part[obj_id]):
            part_relations[obj_id][num_roots + part_id][num_roots + part_id] = -1000000000*np.ones(6)# the parent of the one can't be itself...if so, go to the next one.
            each_parent = np.unravel_index(np.argmax(part_relations[obj_id][num_roots + part_id]), part_relations[obj_id][num_roots + part_id].shape)
            parent_id = each_parent[0]
            relation_id = each_parent[1]
            new_relations[part_id + num_roots, parent_id, relation_id] = 1
        all_new_relations.append(new_relations)
    return all_new_relations

def get_camera_parameters_move(traj_i):
    all_p2s = np.arange(-0.5, 1.5, 0.1)
    p1 = 1.5
    p2 = all_p2s[traj_i]
    c2 = 0.5

    p3 = 1.2
    c3 = 0.5

    view_matrix = p.computeViewMatrix([p1, p2, p3], [0, c2, c3], [0, 0, 1])
    return view_matrix

def traj_camera(view_matrix):
    zfar, znear = 0.01, 10
    fov, aspect, nearplane, farplane = 60, 1, 0.01, 100
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)
    light_pos = [3, 1.5, 5]
    _, _, color, depth, segm= p.getCameraImage(512, 512, view_matrix, projection_matrix, light_pos, shadow=1, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb = np.array(color)[:,:, :3]
    return rgb

def animate(object_id, link_orientations, test_name, headless = False):

    for i in range(20):
        for jid in range(p.getNumJoints(object_id)):
            ji = p.getJointInfo(object_id, jid)
            if ji[16]==-1 and ji[2] == 1:
                jointpos = np.random.uniform(0.2, 0.4)
                p.resetJointState(object_id, jid, jointpos)
            if ji[16]==-1 and ji[2] == 0:
                if link_orientations[int(ji[1][5:])-1][-1] == -1:
                    jointpos = np.random.uniform(0.5, 1)
                elif ji[13][1] == 1:
                    jointpos = np.random.uniform(0.25, 0.7)
                else:
                    jointpos = np.random.uniform(-0.7, -0.25)
                p.resetJointState(object_id, jid, jointpos)
        if headless:
            os.makedirs(f"visualization/{test_name}", exist_ok=True)
            view_matrix = get_camera_parameters_move(i)
            rgb = traj_camera(view_matrix)
            PIL.Image.fromarray(rgb).save(f"visualization/{test_name}/{i}.png")

        time.sleep(0.5)

def process_prediction(part_meshes, part_positions_starts, part_positions_ends, part_relations, base_pred):
    new_part_relations = get_binary_relation_parts(part_relations, part_positions_starts, 1)

    pred_data = {}
    if np.array(base_pred)[0] not in [1,2,3,4,5,7]: # if its not cabinet, shelf, oven, dishwasher, washer and fridge, count as rigid
        part_meshes = []
        part_positions_starts = []
        part_positions_ends = []
        new_part_relations = np.zeros((1,1, 6))
    else:
        part_meshes = np.array(part_meshes)[0]
        part_positions_starts = np.array(part_positions_starts)[0]
        part_positions_ends = np.array(part_positions_ends)[0]
        new_part_relations = np.array(new_part_relations)[0]

    pred_data['part_meshes'] = [part_meshes]
    pred_data['part_positions_start'] = [part_positions_starts]
    pred_data['part_positions_end'] = [part_positions_ends]
    pred_data['part_relations'] = [new_part_relations]
    pred_data['part_bases'] = [np.array(base_pred)[0]]

    return pred_data

def object_prediction(img_path, label_final_dir, urdformer_part, device, with_texture, if_random, headless=False):

    parent_pred_parts = []
    position_pred_end_parts = []
    position_pred_start_parts = []
    mesh_pred_parts = []
    base_types = []

    test_name = os.path.basename(img_path)[:-4]
    image = np.array(PIL.Image.open(img_path).convert("RGB"))
    image_tensor_part, bbox_part, masks_part, tgt_padding_mask_part, tgt_padding_relation_mask_part = evaluate_parts_with_masks(
        f"{label_final_dir}/{test_name}.npy", image)

    position_pred_part, position_pred_end_part, mesh_pred_part, parent_pred_part, base_pred = evaluate_real_image(
        image_tensor_part, bbox_part, masks_part, tgt_padding_mask_part, tgt_padding_relation_mask_part,
        urdformer_part, device)

    size_scale = 4
    scale_pred_part = abs(np.array(size_scale * (position_pred_end_part - position_pred_part) / 12))

    root_position = [0, 0, 0]
    root_orientation = [0, 0, 0, 1]
    root_scale = [1, 1, 1]

    if base_pred[0] == 5:
        root_scale[2]*=2

    scale_pred_part[:, 2] *= root_scale[2]

    ##################################################
    parent_pred_parts.append(np.array(parent_pred_part))
    position_pred_end_parts.append(np.array(position_pred_end_part[:, 1:]))
    position_pred_start_parts.append(np.array(position_pred_part[:, 1:]))
    mesh_pred_parts.append(np.array(mesh_pred_part))
    base_types.append(base_pred[0])

    # visualization
    texture_list = []
    if with_texture:
        ############## load texture if needed ##################
        label_path = f"{label_final_dir}/{test_name}.npy"
        object_info = np.load(label_path, allow_pickle=True).item()
        bboxes = object_info['part_normalized_bbox']

        for bbox_id in range(len(bboxes)):
            if os.path.exists(f"textures/{test_name}/{bbox_id}.png"):
                texture_list.append(f"textures/{test_name}/{bbox_id}.png")
            else:
                print('no texture map found! Run get_texture.py first')


    object_id, link_orientations = visualization_parts(p, root_position, root_orientation, root_scale, base_pred[0],
                                                       position_pred_part, scale_pred_part, mesh_pred_part,
                                                       parent_pred_part, texture_list, if_random, filename=f"output/{test_name}")



    animate(object_id, link_orientations, test_name, headless=headless)

    root = "meshes/cabinet.obj"

    time.sleep(1)

def evaluate(args, with_texture=False, headless = False):
    device = "cuda"
    input_path = args.image_path
    label_dir = "grounding_dino/labels_manual"
    if headless:
        physicsClient = p.connect(p.DIRECT)
    else:
        physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    p.configureDebugVisualizer(1, lightPosition=(1250, 100, 2000), rgbBackground=(1, 1, 1))

    ########################  URDFormer Core  ##############################
    num_relations = 6 # the dimension of the relationship embedding
    urdformer_part = URDFormer(num_relations=num_relations, num_roots=1)
    urdformer_part = urdformer_part.to(device)
    part_checkpoint = "checkpoints/part.pth"
    checkpoint = torch.load(part_checkpoint)
    urdformer_part.load_state_dict(checkpoint['model_state_dict'])

    for img_path in glob.glob(input_path+"/*"):
        p.resetSimulation()
        object_prediction(img_path, label_dir, urdformer_part, device, with_texture, args.random, headless=headless)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--texture', action='store_true', help='adding texture')
    parser.add_argument('--headless', action='store_true', help='option to run in headless mode')
    parser.add_argument('--scene_type', '--scene_type', default='cabinet', type=str)
    parser.add_argument('--image_path', '--image_path', default='images', type=str)
    parser.add_argument('--random', '--random', action='store_true', help='use random meshes from partnet?')

    ##################### IMPORTANT! ###############################
    # URDFormer replies on good bounding boxes of parts and ojects, you can achieve this by our annotation tool (~1min label per image)
    # We also provided our finetuned GroundingDINO (model soup version) to automate/initialize this. We finetuned GroundingDino on our generated dataset, and
    # apply model soup for the pretrained and finetuned GroundingDINO. However, the performance of bbox prediction is not gauranteed and will be our future work.

    args = parser.parse_args()
    evaluate(args, with_texture=args.texture, headless=args.headless)


if __name__ == "__main__":
    main()