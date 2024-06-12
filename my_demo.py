import os
import PIL
import time
import glob
import torch
import argparse
import numpy as np
import pybullet as p
from tqdm import tqdm
from urdformer import URDFormer
import torchvision.transforms as transforms
from utils import my_visualization_parts, viz_graph


# integrate the extracted texture map into URDFormer prediction
def evaluate_real_image(
    image_tensor,
    bbox,
    masks,
    tgt_padding_mask,
    tgt_padding_relation_mask,
    urdformer,
    device,
):
    rgb_input = image_tensor.float().to(device).unsqueeze(0)
    bbox_input = torch.tensor(bbox).float().to(device).unsqueeze(0)
    masks_input = torch.tensor(masks).float().to(device).unsqueeze(0)

    tgt_padding_mask = torch.logical_not(tgt_padding_mask)
    tgt_padding_mask = torch.tensor(tgt_padding_mask).to(device).unsqueeze(0)

    tgt_padding_relation_mask = torch.logical_not(tgt_padding_relation_mask)
    tgt_padding_relation_mask = (
        torch.tensor(tgt_padding_relation_mask).to(device).unsqueeze(0)
    )

    (
        position_x_pred,
        position_y_pred,
        position_z_pred,
        position_x_end_pred,
        position_y_end_pred,
        position_z_end_pred,
        mesh_pred,
        parent_cls,
        base_pred,
    ) = urdformer(rgb_input, bbox_input, masks_input, 2)
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
    position_pred_end = torch.stack(
        [position_pred_x_end, position_pred_y_end, position_pred_z_end]
    ).T

    return (
        position_pred.detach().cpu().numpy(),
        position_pred_end.detach().cpu().numpy(),
        mesh_pred.detach().cpu().numpy(),
        parent_pred.detach().cpu().numpy(),
        base_pred.detach().cpu().numpy(),
    )


def image_transform():
    """Constructs the image preprocessing transform object.

    Arguments:
        image_size (int): Size of the result image
    """
    # ImageNet normalization statistics
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    preprocessing = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

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
    for boxid, each_bbox in enumerate(data["part_normalized_bbox"]):
        bbox.append(each_bbox)
        resized = np.zeros((14, 14))
        resized_mask.append(resized)
    padded_bbox = np.zeros((max_bbox, 4))
    padded_bbox[: len(bbox)] = bbox

    padded_masks = np.zeros((max_bbox, 14, 14))
    padded_masks[: len(resized_mask)] = resized_mask

    tgt_padding_mask = torch.ones([max_bbox])
    tgt_padding_mask[: len(bbox)] = 0.0
    tgt_padding_mask = tgt_padding_mask.bool()

    tgt_padding_relation_mask = torch.ones([max_bbox + num_roots])
    tgt_padding_relation_mask[: len(bbox) + num_roots] = 0.0
    tgt_padding_relation_mask = tgt_padding_relation_mask.bool()

    return (
        image_tensor,
        np.array([padded_bbox]),
        np.array([padded_masks]),
        tgt_padding_mask,
        tgt_padding_relation_mask,
    )


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
    _, _, color, depth, segm = p.getCameraImage(
        512,
        512,
        view_matrix,
        projection_matrix,
        light_pos,
        shadow=1,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    rgb = np.array(color)[:, :, :3]
    return rgb


def animate(
    object_id,
    link_orientations,
    test_name,
    headless=False,
    n_states=3,
    output_path=None,
):
    joint_states = {
        "prismatic": [0, 0.4, 0.8],
        "revolute1": [0, 0.4, 0.8],
        "revolute2": [0, 0.4, 0.8],
        "revolute3": [0, -0.4, -0.8],
    }
    for i in range(n_states):
        for jid in range(p.getNumJoints(object_id)):
            if i == 0:
                perturb = 0
            else:
                perturb = np.random.uniform(-0.2, 0.2)
            ji = p.getJointInfo(object_id, jid)
            if ji[2] == p.JOINT_PRISMATIC:
                jointpos = joint_states["prismatic"][i] + perturb
                p.resetJointState(object_id, jid, jointpos)
            elif ji[2] == p.JOINT_REVOLUTE:
                if link_orientations[int(ji[1][5:]) - 1][-1] == -1:
                    jointpos = joint_states["revolute1"][i] + perturb
                elif ji[13][1] == 1:
                    jointpos = joint_states["revolute2"][i] + perturb
                else:
                    jointpos = joint_states["revolute3"][i] + perturb
                p.resetJointState(object_id, jid, jointpos)
        if headless:
            os.makedirs(f"{output_path}/{test_name}", exist_ok=True)
            view_matrix = get_camera_parameters_move(i)
            rgb = traj_camera(view_matrix)
            PIL.Image.fromarray(rgb).save(f"{output_path}/{test_name}/{i}.png")

        time.sleep(0.5)


def object_prediction(
    img_path,
    output_path,
    label_final_dir,
    urdformer_part,
    device,
    with_texture,
    if_random,
    headless=False,
):

    parent_pred_parts = []
    position_pred_end_parts = []
    position_pred_start_parts = []
    mesh_pred_parts = []
    base_types = []

    test_name = os.path.basename(img_path)[:-4]
    image = np.array(PIL.Image.open(img_path).convert("RGB"))

    (
        image_tensor_part,
        bbox_part,
        masks_part,
        tgt_padding_mask_part,
        tgt_padding_relation_mask_part,
    ) = evaluate_parts_with_masks(f"{label_final_dir}/{test_name}.npy", image)

    (
        position_pred_part,
        position_pred_end_part,
        mesh_pred_part,
        parent_pred_part,
        base_pred,
    ) = evaluate_real_image(
        image_tensor_part,
        bbox_part,
        masks_part,
        tgt_padding_mask_part,
        tgt_padding_relation_mask_part,
        urdformer_part,
        device,
    )

    size_scale = 4
    scale_pred_part = abs(
        np.array(size_scale * (position_pred_end_part - position_pred_part) / 12)
    )

    root_position = [0, 0, 0]
    root_orientation = [0, 0, 0, 1]
    root_scale = [1, 1, 1]

    if base_pred[0] == 5:  # fridge
        root_scale[2] *= 2

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
        bboxes = object_info["part_normalized_bbox"]

        for bbox_id in range(len(bboxes)):
            if os.path.exists(f"textures/{test_name}/{bbox_id}.png"):
                texture_list.append(f"textures/{test_name}/{bbox_id}.png")
            else:
                print("no texture map found! Run get_texture.py first")

    object_id, link_orientations, tree = my_visualization_parts(
        p,
        root_position,
        root_orientation,
        root_scale,
        base_pred[0],
        position_pred_part,
        scale_pred_part,
        mesh_pred_part,
        parent_pred_part,
        texture_list,
        if_random,
        filename=f"output/{test_name}",
    )

    animate(
        object_id,
        link_orientations,
        test_name,
        headless=headless,
        output_path=output_path,
    )

    graph_img = viz_graph(tree, res=256)
    PIL.Image.fromarray(graph_img).save(f"{output_path}/{test_name}/graph.png")

    time.sleep(1)


def evaluate(args, with_texture=False, headless=False):
    device = "cuda"
    input_path = args.image_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    label_dir = "grounding_dino/labels_manual"
    if headless:
        physicsClient = p.connect(p.DIRECT, options="--renderDevice=egl")
    else:
        physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    p.configureDebugVisualizer(
        1, lightPosition=(1250, 100, 2000), rgbBackground=(1, 1, 1)
    )

    ########################  URDFormer Core  ##############################
    num_relations = 6  # the dimension of the relationship embedding
    urdformer_part = URDFormer(num_relations=num_relations, num_roots=1)
    urdformer_part = urdformer_part.to(device)
    part_checkpoint = "checkpoints/part.pth"
    checkpoint = torch.load(part_checkpoint)
    urdformer_part.load_state_dict(checkpoint["model_state_dict"])
    for img_path in tqdm(glob.glob(input_path + "/*")):
        if img_path in ["my_images/val_StorageFurniture_47466_18.png"]:  # buggy output
            # Error msg: corrupted size vs. prev_size
            # from: util.create_articulated_objects, obj = p.createMultiBody (line 75)
            continue
        if img_path in [
            "my_images/val_StorageFurniture_45444_18.png",
            "my_images/test_StorageFurniture_46655_18.png",
            "my_images/val_StorageFurniture_48013_19.png",
            "my_images/val_StorageFurniture_46563_18.png",
            "my_images/test_StorageFurniture_46655_19.png"
        ]:  # buggy output
            # Error msg: IndexError: list index out of range
            # from: link_names[link_id + 1], link_names[linkparents[link_id]] (line 1685, in write_urdf)
            continue
        if os.path.exists(f"{output_path}/{os.path.basename(img_path)[:-4]}"):
            continue
        print(f"Processing {img_path}...")
        p.resetSimulation()
        object_prediction(
            img_path,
            output_path,
            label_dir,
            urdformer_part,
            device,
            with_texture,
            args.random,
            headless=headless,
        )


def collect_html(args):
    html_header = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>URDFormer Qualitative</title>
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid black;
                padding: 8px;
                text-align: left;
            }
            .separator {
                border-top: 2px solid black;
            }
        </style>
    </head>
    <body>
        <table>
    """
    img_dir = args.image_path
    pred_bbox_dir = "grounding_dino/labels_filtered"
    bbox_dir = "grounding_dino/labels_manual"
    out_dir = args.output_path
    html = html_header

    file_list = os.listdir(img_dir)
    file_list.sort()
    for fpath in file_list:
        if fpath.startswith("test") and fpath.endswith(".png"):
            fname = fpath[:-4]
            html += f"""
                    <tr>
                        <th>ID</th>
                        <th>Input Image</th>
                        <th>Predicted Bbox</th>
                        <th>Input GT Bbox</th>
                        <th>Output Graph</th>
                        <th>Output Shape (Animate at 3 States)</th>
                    </tr>
                    <tr>
                        <td style="height: 200px; width: 50px;">{fname}</td>
                        <td><img src="{os.path.join('../', img_dir, fpath)}" alt="Input Image" style="height: 200px; width: auto;">
                        <td><img src="{os.path.join('../', pred_bbox_dir, fname + '.npy.png')}" alt="predicted Bbox" style="height: 200px; width: auto;"></td>
                        <td><img src="{os.path.join('../', bbox_dir, fpath)}" alt="GT Bbox" style="height: 200px; width: auto;"></td>
                        <td><img src="{os.path.join(fname, 'graph.png')}" alt="Graph" style="height: 200px; width: auto;"></td>
                        <td>
                        <img src="{os.path.join(fname, '0.png')}" alt="Generated Item" style="height: 200px; width: auto;">
                        <img src="{os.path.join(fname, '1.png')}" alt="Generated Item" style="height: 200px; width: auto;">
                        <img src="{os.path.join(fname, '2.png')}" alt="Generated Item" style="height: 200px; width: auto;">
                        </td>
                    </tr>
                    <tr class="separator"><td colspan="3"></td></tr>
                    """

    html += "</table></body></html>"
    with open(os.path.join(out_dir, f"test.html"), "w") as f:
        f.write(html)

    html = html_header
    for fpath in file_list:
        if fpath.startswith("val") and fpath.endswith(".png"):
            fname = fpath[:-4]
            html += f"""
                    <tr>
                        <th>ID</th>
                        <th>Input Image</th>
                        <th>Predicted Bbox</th>
                        <th>Input GT Bbox</th>
                        <th>Output Graph</th>
                        <th>Output Shape (Animate at 3 States)</th>
                    </tr>
                    <tr>
                        <td style="height: 200px; width: 50px;">{fname}</td>
                        <td><img src="{os.path.join('../', img_dir, fpath)}" alt="Input Image" style="height: 200px; width: auto;">
                        <td><img src="{os.path.join('../', pred_bbox_dir, fname + '.npy.png')}" alt="predicted Bbox" style="height: 200px; width: auto;"></td>
                        <td><img src="{os.path.join('../', bbox_dir, fpath)}" alt="GT Bbox" style="height: 200px; width: auto;"></td>
                        </td><img src="{os.path.join(fname, 'graph.png')}" alt="Graph" style="height: 200px; width: auto;"></td>
                        <td>
                        <img src="{os.path.join(fname, '0.png')}" alt="Generated Item" style="height: 200px; width: auto;">
                        <img src="{os.path.join(fname, '1.png')}" alt="Generated Item" style="height: 200px; width: auto;">
                        <img src="{os.path.join(fname, '2.png')}" alt="Generated Item" style="height: 200px; width: auto;">
                        </td>
                    </tr>
                    <tr class="separator"><td colspan="3"></td></tr>
                    """

    html += "</table></body></html>"

    with open(os.path.join(out_dir, f"val.html"), "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--texture", action="store_true", help="adding texture")
    parser.add_argument(
        "--headless", action="store_true", help="option to run in headless mode"
    )
    parser.add_argument("--scene_type", "--scene_type", default="cabinet", type=str)
    parser.add_argument("--image_path", "--image_path", default="my_images", type=str)
    parser.add_argument(
        "--output_path", "--output_path", default="my_visualization", type=str
    )
    parser.add_argument(
        "--random",
        "--random",
        action="store_true",
        help="use random meshes from partnet?",
    )

    ##################### IMPORTANT! ###############################
    # URDFormer replies on good bounding boxes of parts and ojects, you can achieve this by our annotation tool (~1min label per image)
    # We also provided our finetuned GroundingDINO (model soup version) to automate/initialize this. We finetuned GroundingDino on our generated dataset, and
    # apply model soup for the pretrained and finetuned GroundingDINO. However, the performance of bbox prediction is not gauranteed and will be our future work.

    args = parser.parse_args()
    evaluate(args, with_texture=args.texture, headless=args.headless)

    collect_html(args)


if __name__ == "__main__":
    main()
