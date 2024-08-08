import os
import json
import imageio
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
import xml.etree.ElementTree as ET
from my_utils import get_hash, prepare_meshes, draw_boxes_axiss_anim

# open3d verbosity
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def _rpy_to_matrix(roll, pitch, yaw):
    R_x = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]],
        dtype=np.float32,
    )
    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=np.float32,
    )
    R_z = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def _get_origin(element):
    origin = element.find("origin")
    if origin is not None:
        xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
        return np.array(xyz, dtype=np.float32), _rpy_to_matrix(*rpy)
    return np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32)


def _get_joint_limit(joint, joint_type):
    limit = joint.find("limit")
    if limit is not None:
        if joint_type == "revolute":
            lower = np.rad2deg(float(limit.attrib.get("lower", 0.0)))
            upper = np.rad2deg(float(limit.attrib.get("upper", 0.0)))
            return [0.0, upper - lower]
        elif joint_type == "prismatic":
            lower = float(limit.attrib.get("lower", 0.0))
            upper = float(limit.attrib.get("upper", 0.0))
            return [0.0, upper - lower]
        elif joint_type == "continuous":
            return [0.0, 0.0]
        elif joint_type == "fixed":
            return [0.0, 0.0]
        else:
            raise ValueError("Unknown joint type: {}".format(joint_type))


def apply_transform(position, orientation, translation, rotation):
    new_position = np.dot(orientation, position) + translation
    new_orientation = np.dot(orientation, rotation)
    return new_position, new_orientation


def _find_base_link(root):
    # find the link with the name 'base_link'
    for link in root.findall("link"):
        if link.attrib["name"] == "base_link":
            return link


def _get_aabb_info(root, link_name, translation, rotation, save_mesh_path=None):
    mesh_path, mesh_scale = _get_mesh(root, link_name)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    scaling = np.eye(4, dtype=np.float32)
    scaling[0, 0] = mesh_scale[0]
    scaling[1, 1] = mesh_scale[1]
    scaling[2, 2] = mesh_scale[2]
    mesh.transform(scaling)
    transformation = np.eye(4, dtype=np.float32)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = translation
    mesh.transform(transformation)
    aabb = mesh.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    size = aabb.get_max_bound() - aabb.get_min_bound()
    o3d.io.write_triangle_mesh(save_mesh_path, mesh)
    return center, size


def _get_mesh(root, link_name):
    mesh_element = (
        root.find(f'link[@name="{link_name}"]')
        .find("visual")
        .find("geometry")
        .find("mesh")
    )
    mesh_path = mesh_element.attrib["filename"][9:]  # remove '../../' prefix
    mesh_scale = np.array(
        [float(v) for v in mesh_element.attrib["scale"].split()], dtype=np.float32
    )
    return mesh_path, mesh_scale


def get_joint_transform(joint):
    xyz, rpy_matrix = _get_origin(joint)
    axis_element = joint.find("axis")
    if axis_element is not None:
        axis = np.array(
            [float(v) for v in axis_element.attrib["xyz"].split()], dtype=np.float32
        )
    else:
        axis = np.array([0, 0, 1], dtype=np.float32)  # default axis if not specified
    return xyz, rpy_matrix, axis


def urdf_to_json(root_dir):
    assert os.path.exists(os.path.join(root_dir, "object.urdf")) 
    # create the directory to store the ply files
    os.makedirs(os.path.join(root_dir, "plys"), exist_ok=True)
    # initialize json data to store the object information
    json_data = {
        "meta": {},
        "diffuse_tree": [],
    }
    # global information of the object
    global_info = {}
    # load urdf file
    root = ET.parse(os.path.join(root_dir, "object.urdf")).getroot()
    # get base link
    base_link = _find_base_link(root)
    base_link_name = base_link.attrib["name"]
    # to keep track of the visited links
    visited = [base_link_name]
    # get the position and orientation of the base link
    base_pos, base_rot = _get_origin(base_link.find("visual"))
    global_info[base_link_name] = {
        "position": base_pos,
        "orientation": base_rot,
    }
    # get the aabb info of the base link and save part mesh
    base_aabb_center, base_aabb_size = _get_aabb_info(
        root,
        base_link_name,
        base_pos,
        base_rot,
        save_mesh_path=os.path.join(root_dir, "plys/base.ply"),
    )
    # register the base node
    json_data["diffuse_tree"].append(
        {
            "id": 0,
            "name": "base",
            "parent": -1,
            "children": [],
            "joint": {
                "type": "fixed",
                "range": [0, 0],
                "axis": {"origin": [0, 0, 0], "direction": [0, 0, 0]},
            },
            "aabb": {
                "center": base_aabb_center.tolist(),
                "size": base_aabb_size.tolist(),
            },
            "plys": ["plys/base.ply"],
        }
    )

    # get all the joints
    joints = root.findall("joint")
    # iterate over the joints to build the tree
    for _ in range(len(joints)):
        for joint in joints:
            parent_name = joint.find("parent").attrib["link"]
            child_name = joint.find("child").attrib["link"]
            # to make sure the parent is visited before the child
            if parent_name in visited and child_name not in visited:
                # mark the child as visited
                visited.append(child_name)
                parent_id = visited.index(parent_name)
                child_id = visited.index(child_name)
                # get the global position and orientation of the parent link
                parent_pos, parent_ori = (
                    global_info[parent_name]["position"],
                    global_info[parent_name]["orientation"],
                )
                # get the global position and orientation of the child link
                child_T, child_R, child_axis = get_joint_transform(joint) # load local info
                child_pos, child_ori = apply_transform(
                    parent_pos, parent_ori, child_T, child_R
                )
                # get the joint type and axis direction
                joint_type = joint.attrib["type"]
                global_axis_dir = np.dot(parent_ori, child_axis)
                # get the aabb info of the child link and save part mesh
                aabb_center, aabb_size = _get_aabb_info(
                    root, child_name, child_pos, child_ori, 
                    save_mesh_path = os.path.join(root_dir, "plys", f"{child_name}.ply")
                )
                # get the global axis origin
                global_axis_ori = child_pos
                if joint_type in ["prismatic", "fixed"]:
                    global_axis_ori = aabb_center
                # register the child node
                json_data["diffuse_tree"].append(
                    {
                        "id": child_id,
                        "name": child_name,
                        "parent": parent_id,
                        "children": [],
                        "joint": {
                            "type": joint_type,
                            "range": _get_joint_limit(joint, joint_type),
                            "axis": {
                                "origin": global_axis_ori.tolist(),
                                "direction": (
                                    global_axis_dir.tolist()
                                    if joint_type != "fixed"
                                    else [0, 0, 0]
                                ),
                            },
                        },
                        "aabb": {
                            "center": aabb_center.tolist(),
                            "size": aabb_size.tolist(),
                        },
                        "plys": [f"plys/{child_name}.ply"],
                    }
                )
                # update the connectivity of the parent node
                json_data["diffuse_tree"][parent_id]["children"].append(child_id)
                # update the global information
                global_info[child_name] = {
                    "position": child_pos,
                    "orientation": child_ori,
                }


    # update the meta information
    json_data["meta"]["tree_hash"] = get_hash(json_data)
    json_data["meta"]["obj_cat"] = root_dir.split("_")[0]

    return json_data


def _zero_center_object(json_data, translation):
    """
    Zero center the object as a whole\n
    """
    for part in json_data["diffuse_tree"]:
        part["aabb"]["center"] = (
            np.array(part["aabb"]["center"], dtype=np.float32) + translation
        ).tolist()
        if part["joint"]["type"] != "fixed":
            part["joint"]["axis"]["origin"] = (
                np.array(part["joint"]["axis"]["origin"], dtype=np.float32)
                + translation
            ).tolist()


def _rotate_object(json_data, rotation):
    """
    Rotate the object as a whole\n
    """
    for part in json_data["diffuse_tree"]:
        part["aabb"]["center"] = np.dot(
            rotation, np.array(part["aabb"]["center"], dtype=np.float32)
        ).tolist()
        part["aabb"]["size"] = np.dot(
            rotation, np.array(part["aabb"]["size"], dtype=np.float32)
        ).tolist()
        if part["joint"]["type"] != "fixed":
            part["joint"]["axis"]["origin"] = np.dot(
                rotation, np.array(part["joint"]["axis"]["origin"])
            ).tolist()
            part["joint"]["axis"]["direction"] = np.dot(
                rotation, np.array(part["joint"]["axis"]["direction"])
            ).tolist()


def _get_zero_center_translation(json_data):
    """
    Compute the translation to zero center the object
    """
    bbox_min = np.zeros((len(json_data["diffuse_tree"]), 3), dtype=np.float32)
    bbox_max = np.zeros((len(json_data["diffuse_tree"]), 3), dtype=np.float32)
    # For each part, compute the bounding box and store the min and max vertices
    for part_idx, part in enumerate(json_data["diffuse_tree"]):
        bbox_center = np.array(part["aabb"]["center"], dtype=np.float32)
        bbox_size_half = np.array(part["aabb"]["size"], dtype=np.float32) / 2
        bbox_min[part_idx] = bbox_center - bbox_size_half
        bbox_max[part_idx] = bbox_center + bbox_size_half
    # Compute the overall bounding box size and center
    bbox_min = np.min(bbox_min, axis=0)
    bbox_max = np.max(bbox_max, axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    # Compute the translation to zero center the object
    translation = -bbox_center
    return translation


def _recoordinate_meshes(json_data, translation, rotation, root_dir):
    tree = json_data["diffuse_tree"]
    for node in tree:
        fname = node["plys"][0]
        mesh = o3d.io.read_triangle_mesh(os.path.join(root_dir, fname))
        mesh.compute_vertex_normals()
        mesh.translate(translation)
        mesh.rotate(rotation, center=(0, 0, 0))
        o3d.io.write_triangle_mesh(os.path.join(root_dir, fname), mesh)


def recoordinate_object(root_dir, json_data):
    # zero center the object
    translation = _get_zero_center_translation(json_data)
    _zero_center_object(json_data, translation)
    # rotate the object from z-up, x-front to y-up, z-front
    rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
    _rotate_object(json_data, rotation)
    # re-coordinate the meshes
    _recoordinate_meshes(json_data, translation, rotation, root_dir)

def convert(root_dir):
    json_data = urdf_to_json(root_dir)
    recoordinate_object(root_dir, json_data)
    with open(os.path.join(root_dir, "object.json"), "w") as f:
        json.dump(json_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="exps/freezed_pred", type=str)

    args = parser.parse_args()

    cases = os.listdir(args.exp_dir)
    cases.remove('collect.html')
    cases.sort()
    for case in tqdm(cases):
        if not os.path.exists(os.path.join(args.exp_dir, case, "object.json")):
            try:
                convert(os.path.join(args.exp_dir, case))
            except Exception as e:
                with open('convert_err.log', 'a') as f:
                    f.write(f"{case}: {e}\n")


    # [debug & verfiy] visualize the object
    # viz_meshes = prepare_meshes(json_data)
    # viz_img = draw_boxes_axiss_anim(
    #     viz_meshes["bbox_0"],
    #     viz_meshes["bbox_1"],
    #     viz_meshes["axiss"],
    #     resolution=128,
    # )

    # imageio.imwrite('test.jpg', viz_img)
