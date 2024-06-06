import pybullet as p
import time
import pybullet_data
import numpy as np
import random
import glob
import torch
from PIL import Image
import PIL
from scipy.spatial.transform import Rotation as Rot
import json
import cv2
import torchvision.transforms as transforms
import os
from argparse import ArgumentParser
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO

# np.random.seed(777) #77, 777, 24
# ref_colors = np.random.uniform(low=0.6, high=0.9, size=(32, 3))
ref_colors = np.array([[0.64579912, 0.69070698, 0.61861092],
       [0.7379581 , 0.85057602, 0.87809911],
       [0.8180967 , 0.83054887, 0.68076152],
       [0.79320879, 0.62801198, 0.62390577],
       [0.77688413, 0.70300216, 0.89666284],
       [0.78794196, 0.80453378, 0.76567704],
       [0.68065802, 0.71197782, 0.66687843],
       [0.65593278, 0.71719443, 0.65794872],
       [0.78327328, 0.86484253, 0.78670165],
       [0.67593568, 0.65397909, 0.84492134],
       [0.66761149, 0.75505714, 0.75554875],
       [0.78011248, 0.75978614, 0.60399301],
       [0.75722918, 0.86876541, 0.83097039],
       [0.63685551, 0.68876181, 0.78360707],
       [0.81784144, 0.73904924, 0.83073311],
       [0.65748931, 0.76736002, 0.76523345],
       [0.74166765, 0.83756549, 0.6345749 ],
       [0.80439117, 0.70870008, 0.70326267],
       [0.73485562, 0.60808268, 0.72457431],
       [0.87666951, 0.62736167, 0.69453653],
       [0.75840667, 0.69841861, 0.73467466],
       [0.60490032, 0.62910807, 0.80777657],
       [0.85078302, 0.7272966 , 0.85463229],
       [0.76403736, 0.70623104, 0.8181749 ],
       [0.6281555 , 0.86785764, 0.70087748],
       [0.8675498 , 0.6890547 , 0.69049449],
       [0.84187218, 0.85128299, 0.7902844 ],
       [0.69339819, 0.60883457, 0.71993319],
       [0.75545204, 0.60221654, 0.83248433],
       [0.85634136, 0.63945985, 0.68630209],
       [0.69797664, 0.87196787, 0.89986786],
       [0.83126529, 0.61285426, 0.88957533]])

# ref_colors = np.array([[0.4980392156862745, 0.23529411764705882, 0.5529411764705883], [0.06666666666666667, 0.6470588235294118, 0.4745098039215686], [0.2235294117647059, 0.4117647058823529, 0.6745098039215687], [0.9490196078431372, 0.7176470588235294, 0.00392156862745098], [0.9058823529411765, 0.24705882352941178, 0.4549019607843137], [0.5019607843137255, 0.7294117647058823, 0.35294117647058826], [0.9019607843137255, 0.5137254901960784, 0.06274509803921569], [0.0, 0.5254901960784314, 0.5843137254901961], [0.8117647058823529, 0.10980392156862745, 0.5647058823529412], [0.9764705882352941, 0.4823529411764706, 0.4470588235294118], [0.6470588235294118, 0.6666666666666666, 0.6], [0.37254901960784315, 0.27450980392156865, 0.5647058823529412], [0.11372549019607843, 0.4117647058823529, 0.5882352941176471], [0.2196078431372549, 0.6509803921568628, 0.6470588235294118], [0.058823529411764705, 0.5215686274509804, 0.32941176470588235], [0.45098039215686275, 0.6862745098039216, 0.2823529411764706], [0.9294117647058824, 0.6784313725490196, 0.03137254901960784], [0.8823529411764706, 0.48627450980392156, 0.0196078431372549], [0.8, 0.3137254901960784, 0.24313725490196078], [0.5803921568627451, 0.20392156862745098, 0.43137254901960786], [0.43529411764705883, 0.25098039215686274, 0.4392156862745098], [0.4, 0.4, 0.4]])

def viz_graph(tree, res=256):
    '''
    Function to plot the directed graph

    Args:
    - info_dict (dict): output json containing the graph information
    - res (int): resolution of the image

    Returns:
    - img_arr (np.array): image array
    '''
    def get_color(n_colors):
        colors = [[0.8, 0.8, 0.8]]
        for i in range(n_colors - 1):
            colors.append(ref_colors[i])
        return colors
    # build tree
    edges = []
    for node in tree:
        edges += [(node['id'], child) for child in node['children']]
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # plot tree
    plt.figure(figsize=(res/100, res/100))

    colors = get_color(len(tree))
    pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")
    node_order = sorted(G.nodes())
    nx.draw(G, pos, node_color=colors, nodelist=node_order, edge_color='k', with_labels=False)
    
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    img_arr = np.asarray(img)
    buf.close()
    plt.clf()
    plt.close()
    return img_arr[:, :, :3]

def visual_collision_shapes(links, link_scales, link_positions, random_color=True):
    collisionIndices, visualIndices = [], []
    if random_color:
        color = [
            np.random.uniform(0.6, 0.9),
            np.random.uniform(0.6, 0.9),
            np.random.uniform(0.6, 0.9),
            1,
        ]
    for i, each_link in enumerate(links):

        if "handle" in each_link or "knob" in each_link:
            color = [0.1, 0.1, 0.1, 1]
            link_scales[i] = [1, 1, 1]
        
        if not random_color:
            color = [ref_colors[i][0], ref_colors[i][1], ref_colors[i][2], 1]

        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=each_link,
            rgbaColor=[color[0], color[1], color[2], color[3]],
            specularColor=[0.5, 0.4, 0],
            meshScale=link_scales[i],
        )
        collisionShapeId = p.createCollisionShape(
            shapeType=p.GEOM_MESH, fileName=each_link, meshScale=link_scales[i]
        )

        visualIndices.append(visualShapeId)
        collisionIndices.append(collisionShapeId)

    return collisionIndices, visualIndices, link_positions



def domain_randomization(
    root_name, link_names, random_frame, linkJointAxis, jointtypes
):
    # ransomize assets
    asset_path = "meshes/partnet_uv"

    new_drawer = random.choice(glob.glob(asset_path + "/drawers/*.obj"))
    new_doorL = random.choice(glob.glob(asset_path + "/doorLs/*.obj"))
    new_doorR = asset_path + "/doorRs/{}".format(os.path.basename(new_doorL))
    if random_frame:
        root_name = random.choice(glob.glob(asset_path + "/cabinet_frames/*.obj"))
    # pick a random handle
    new_handle = random.choice(
        glob.glob(asset_path + "/{}/*.obj".format(random.choice(["handles"])))
    )
    new_links = link_names.copy()

    for link_id, each_name in enumerate(link_names):
        if "drawer" in each_name:
            new_links[link_id] = new_drawer
        if "doorL" in each_name:
            swap = new_drawer
            new_links[link_id] = swap
            if new_drawer == swap:
                linkJointAxis[link_id] = [1, 0, 0]
                jointtypes[link_id] = 1
        if "doorR" in each_name:
            new_links[link_id] = new_doorR
        if "handle" in each_name or "knob" in each_name:
            new_links[link_id] = new_handle

    return root_name, new_links, linkJointAxis, jointtypes


def create_articulated_objects(
    root,
    root_scale,
    root_position,
    root_orientation,
    links,
    link_scales,
    link_positions,
    link_orientations,
    linkparents,
    jointtypes,
    linkJointAxis,
    texture_list,
    random,
):
    if random:
        random_frame = True
        root, links, linkJointAxis, jointtypes = domain_randomization(
            root, links, random_frame, linkJointAxis, jointtypes
        )
    # [EDIT by ljy] not using random color
    collisionIndices, visualIndices, link_positions = visual_collision_shapes(
        links, link_scales, link_positions, random_color=False
    )
    obj = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(
            shapeType=p.GEOM_MESH, fileName=root, meshScale=root_scale
        ),
        baseVisualShapeIndex=p.createVisualShape(
            shapeType=p.GEOM_MESH, fileName=root, meshScale=root_scale
        ),
        basePosition=root_position,
        baseOrientation=root_orientation,
        linkMasses=[0.2] * len(jointtypes),
        linkCollisionShapeIndices=collisionIndices,
        linkVisualShapeIndices=visualIndices,
        linkPositions=link_positions,
        linkOrientations=link_orientations,
        linkInertialFramePositions=[[0, 0, 0]] * len(jointtypes),
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * len(jointtypes),
        linkParentIndices=linkparents,
        linkJointTypes=jointtypes,
        linkJointAxis=linkJointAxis,
    )

    if "meshes/oven_fan.obj" in root:
        color = [0.3, 0.3, 0.3, 1]
        p.changeVisualShape(obj, -1, rgbaColor=color)
    if len(texture_list) > 0:
        if "meshes/oven.obj" in root:
            base_texture = "default_textures/inside.jpg"
        else:
            base_texture = os.path.dirname(texture_list[0]) + "/base.png"

        base_tex = p.loadTexture(base_texture)
        p.changeVisualShape(obj, -1, rgbaColor=(1, 1, 1, 1), textureUniqueId=base_tex)
    for i in range(p.getNumJoints(obj)):
        jointinfo = p.getJointInfo(obj, i)
        if len(texture_list) > 0:
            cab_texture = texture_list[int(jointinfo[12][4:]) - 1]
            cab_tex = p.loadTexture(cab_texture)
            p.changeVisualShape(obj, i, rgbaColor=(1, 1, 1, 1), textureUniqueId=cab_tex)

    return obj


def detection_config(args):
    detection_args = {}
    detection_args["inputs"] = args.image_path
    detection_args["model"] = (
        "grounding_dino/configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_urdformer.py"
    )
    detection_args["weights"] = "grounding_dino/object_souped.pth"
    detection_args["texts"] = (
        "drawer . cabinet_door . handle . knob . right_door . left_door"
    )
    detection_args["device"] = "cuda:0"
    detection_args["pred_score_thr"] = 0.3
    detection_args["batch_size"] = 1
    detection_args["show"] = False
    detection_args["no_save_vis"] = False
    detection_args["no_save_pred"] = False
    detection_args["print_result"] = False
    detection_args["palette"] = "none"
    detection_args["custom_entities"] = False
    detection_args["out_dir"] = "grounding_dino/labels"

    return detection_args


def visualization_global(
    p, mesh_pred, position_pred, position_pred_end, scale_pred, parent_pred, if_random
):
    base_path = "meshes/layout"
    root_paths = ["floor", "ceiling", "front_wall", "left_wall", "right_wall"]
    p.resetSimulation()

    test_links = [[], [], [], [], []]
    test_link_scales = [[], [], [], [], []]
    test_link_positions = [[], [], [], [], []]
    test_link_orientations = [[], [], [], [], []]
    test_link_parents = [[], [], [], [], []]
    test_roots = []
    for root in root_paths:
        test_roots.append(base_path + "/" + str(root) + ".obj")
    num_roots = len(root_paths)
    front_object_position_end = []
    for mesh_id, each_mesh in enumerate(mesh_pred):
        each_parent = np.unravel_index(
            np.argmax(parent_pred[num_roots + mesh_id]),
            parent_pred[num_roots + mesh_id].shape,
        )
        parent_id = each_parent[0]
        cube_path = "meshes/cubes" + str(root_paths[parent_id]) + "/cube.obj"

        relation_id = np.argmax(parent_pred[mesh_id + num_roots, parent_id])

        test_links[parent_id].append(cube_path)

        test_link_orientations[parent_id].append([0, 0, 0, 1])
        front_object_position_end.append(position_pred_end[mesh_id][1])

        test_link_scales[parent_id].append(scale_pred[mesh_id])
        test_link_positions[parent_id].append(position_pred[mesh_id])
        test_link_parents[parent_id].append(0)

    for root_id, each_root in enumerate(test_roots):

        jointtypes, linkJointAxis = [p.JOINT_FIXED] * len(test_links[root_id]), [
            [0, 0, 1]
        ] * len(test_links[root_id])

        texUid = p.loadTexture("textures/texture.png")

        root_position = [0, 0, 0]
        if root_id == 4:
            right_wall_distance = max(front_object_position_end)
            root_position = [0, min(12, right_wall_distance + random.choice([1, 2])), 0]
        obj = create_articulated_objects(
            each_root,
            [1, 1, 1],
            root_position,
            [0, 0, 0, 1],
            test_links[root_id],
            test_link_scales[root_id],
            test_link_positions[root_id],
            test_link_orientations[root_id],
            test_link_parents[root_id],
            jointtypes,
            linkJointAxis,
            texUid,
            if_random,
        )


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
    # rgb = np.array(color)[:, :, :3]
    rgb = np.array(color)

    # Get depth image.
    depth_image_size = (512, 512)
    zbuffer = np.array(depth).reshape(depth_image_size)
    depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
    depth = (2.0 * znear * zfar) / depth

    return rgb, depth, segm


def get_camera_parameters():
    p1 = 2.2
    p2 = -0.5
    c2 = 0.3

    p3 = 0.8
    c3 = 0.8
    view_matrix = p.computeViewMatrix([p1, p2, p3], [0, c2, c3], [0, 0, 1])

    return view_matrix


def manual_label(img, bbox):
    def combine_bbox(image, bounding_boxes):
        selected_boxes = set()
        clicked_box_index = None

        def combine_selected_boxes(bounding_boxes, selected_boxes):
            if len(selected_boxes) != 2:
                return

            box1, box2 = [bounding_boxes[i] for i in selected_boxes]

            x1 = min(box1[0], box2[0])
            y1 = min(box1[1], box2[1])
            x2 = max(box1[2], box2[2])
            y2 = max(box1[3], box2[3])
            combined_box = (x1, y1, x2, y2)
            bounding_boxes.append(combined_box)

        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_boxes
            nonlocal clicked_box_index
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if the user clicked inside any bounding box
                clicked_box_index = None
                for i, box in enumerate(bounding_boxes):
                    y1, x1, y2, x2 = box
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        clicked_box_index = i
                        break

                if clicked_box_index is not None:
                    if flags & cv2.EVENT_FLAG_SHIFTKEY:
                        # Shift key is held down, add to selected boxes
                        selected_boxes.add(clicked_box_index)
                    else:
                        # Shift key is not held down, clear selected boxes and select this box
                        selected_boxes = {clicked_box_index}

                    if len(selected_boxes) == 2:
                        # Combine selected boxes if exactly two boxes are selected
                        combine_selected_boxes(bounding_boxes, selected_boxes)
                        selected_boxes.clear()

        # Create a window and set the mouse callback function
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", mouse_callback)
        colors = np.random.random((len(bounding_boxes) + 50, 3)) * 200
        while True:
            # Create a copy of the original image
            display_image = image.copy()
            # # Highlight selected boxes in a different color
            # for i, box in enumerate(bounding_boxes):
            #     if i in selected_boxes:
            #         color = (255, 0, 0)  # Red for selected boxes
            #     else:
            #         color = (0, 255, 0)  # Green for unselected boxes
            #     y1, x1, y2, x2 = box
            #     cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)

            # Display all bounding boxes
            for i, box in enumerate(bounding_boxes):
                y1, x1, y2, x2 = box
                cv2.rectangle(display_image, (x1, y1), (x2, y2), colors[i], 2)

            cv2.imshow("Image", display_image)
            key = cv2.waitKey(1)
            if key == 27:  # Press Esc to exit
                break
            if key == ord("d") and clicked_box_index is not None:
                for each_bbox in selected_boxes:
                    bounding_boxes.pop(each_bbox)

                for i, box in enumerate(bounding_boxes):
                    y1, x1, y2, x2 = box
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), colors[i], 2)
                cv2.imshow("Image", display_image)

        cv2.destroyAllWindows()

        return bounding_boxes

    def remove_bbox(image, bounding_boxes):
        highlighted_box_index = None
        deleted_boxes = []

        def mouse_callback(event, x, y, flags, param):
            nonlocal highlighted_box_index

            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if the user clicked inside any bounding box
                for i, box in enumerate(bounding_boxes):
                    y1, x1, y2, x2 = box
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        highlighted_box_index = i
                        break
                else:
                    highlighted_box_index = None

        # Create a window and set the mouse callback function
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", mouse_callback)
        colors = np.random.random((len(bounding_boxes), 3)) * 200
        while True:
            # Create a copy of the original image
            display_image = image.copy()

            # Display all bounding boxes
            for i, box in enumerate(bounding_boxes):
                y1, x1, y2, x2 = box
                cv2.rectangle(display_image, (x1, y1), (x2, y2), colors[i], 2)

            cv2.imshow("Image", display_image)

            key = cv2.waitKey(1)
            if key == 27:  # Press Esc to exit
                break

            if key == ord("d") and highlighted_box_index is not None:
                # Delete the highlighted box if the user presses "d"
                deleted_box = bounding_boxes.pop(highlighted_box_index)
                deleted_boxes.append(deleted_box)
                highlighted_box_index = None
                # Re-display the updated image
                display_image = image.copy()
                for box in bounding_boxes:
                    y1, x1, y2, x2 = box
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow("Image", display_image)

            if key == ord("u") and deleted_boxes:
                # Undo the deletion if the user presses "u"
                restored_box = deleted_boxes.pop()
                bounding_boxes.append(restored_box)
                display_image = image.copy()
                for box in bounding_boxes:
                    y1, x1, y2, x2 = box
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow("Image", display_image)

        cv2.destroyAllWindows()
        return bounding_boxes

    # get the initial bbox
    def draw_rectangle(event, x, y, flags, param):
        global ix, iy, drawing, img, img_temp
        # When the left mouse button is pressed, record the starting (x, y) coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        # When moving the mouse, with the left button pressed, draw the rectangle
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                img_temp = img.copy()
                cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow("Image", img_temp)
        # When the left mouse button is released, stop drawing
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", img)

    # Load the image
    img = cv2.imread("path_to_your_image.jpg")
    cv2.imshow("Image", img)

    # Create a temporary image for preview purposes
    img_temp = img.copy()

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_rectangle)

    # Initialize variables
    drawing = False
    ix, iy = -1, -1

    # Keep the window open until a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def my_visualization_parts(
    p,
    root_position,
    root_orientation,
    root_scale,
    mesh_base,
    position_pred_ori,
    scale_pred_ori,
    mesh_pred_ori,
    parent_pred_ori,
    texture_list,
    if_random,
    filename="output",
):
    num_parts = len(mesh_pred_ori)
    num_roots = 1

    part_names = [
        "none",
        "drawer",
        "doorL",
        "doorR",
        "handle",
        "knob",
        "washer_door",
        "doorD",
        "oven_door",
        "doorU",
    ]
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
    part_path = "meshes/parts/"

    # fix the oven fan problem: if there are parts, root shouldn't be oven fan
    if len(position_pred_ori) > 0 and mesh_base == 6:
        mesh_base = 1

    root = "meshes/{}.obj".format(base_names[mesh_base])

    position_type = np.arange(13) / 12
    links = []
    link_scales = []
    link_positions = []
    link_orientations = []
    linkparents = []
    jointtypes = []
    linkJointAxis = []

    parent_pred = []
    relations_pred = []

    if mesh_base >= 9:  # rigid objects
        object_path = "meshes/{}.obj".format(base_names[mesh_base])
        obj = create_obj(p, object_path, root_scale, root_position, root_orientation)
        p.changeVisualShape(obj, -1, rgbaColor=[0.6, 0.6, 0.6, 1])
        # if base_names[mesh_base]=="square_table":
        if mesh_base > 9:
            base_texture = random.choice(glob.glob("default_textures/cab_wood/*"))
            base_tex = p.loadTexture(base_texture)
            p.changeVisualShape(
                obj, -1, rgbaColor=(1, 1, 1, 1), textureUniqueId=base_tex
            )
    else:
        for i, each_parent in enumerate(parent_pred_ori[num_roots:]):
            each_parent = np.unravel_index(np.argmax(each_parent), each_parent.shape)
            parent_id = each_parent[0]
            # [EDIT by ljy] if invalid parent prediction
            if parent_id >= num_parts:
                parent_id = 0
            #########################################
            parent_pred.append(parent_id)
            relations_pred.append(each_parent[1])

        parent_pred = np.array(parent_pred)
        mesh_pred = np.array(mesh_pred_ori)
        scale_pred = np.array(scale_pred_ori)
        position_pred = np.array(position_pred_ori)

        # reorder the parts in ascending order according to the parent id
        new_order = np.argsort(parent_pred)  # ascending order
        parent_pred, scale_pred, mesh_pred, position_pred = (
            parent_pred[new_order],
            scale_pred[new_order],
            mesh_pred[new_order],
            position_pred[new_order],
        )
        if len(texture_list) > 0:
            texture_list = [texture_list[i] for i in new_order]

        # update parent labels
        for p_i, each_parent in enumerate(parent_pred):
            if each_parent > 0:
                if each_parent - 1 in list(new_order):
                    parent_pred[p_i] = list(new_order).index(each_parent - 1) + 1

        parents = {}
        children = {}
        # build the tree structure for visualization
        for i, each_parent in enumerate(parent_pred):
            parent_id = each_parent
            if i + 1 not in parents.keys():
                parents.update({i + 1: parent_id})
            if parent_id not in children.keys():
                children.update({parent_id: [i + 1]})
            else:
                children[parent_id].append(i + 1)
        
        tree = []
        for i in range(len(parents) + 1):
            tree.append({
                "id": i ,
                "children": [] if i not in children.keys() else children[i],
                "parent": parents[i] if i in parents.keys() else -1,
            })

        for i, each_parent in enumerate(parent_pred):
            each_mesh = int(mesh_pred[i])
            parent_id = each_parent

            if each_mesh == 1:  # drawer
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append(
                    [1, scale_pred[i][1] - 0.02, scale_pred[i][2] - 0.02]
                )
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1].astype(int)] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, 1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_PRISMATIC)
                linkJointAxis.append([1, 0, 0])
                
            elif each_mesh == 2:  # left door
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append(
                    [1, scale_pred[i][1] - 0.01, scale_pred[i][2] - 0.01]
                )
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1].astype(int)] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, 1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 0, 1])
            elif each_mesh == 3:  # right door
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append(
                    [1, scale_pred[i][1] - 0.01, scale_pred[i][2] - 0.01]
                )
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1].astype(int)] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, -1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 0, 1])
            elif each_mesh == 6:  # washerdoor
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1].astype(int)] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, 1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 0, 1])
            elif each_mesh == 7:  # down door
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1].astype(int)] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, 1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 1, 0])
            elif each_mesh == 8:  # oven door:
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1].astype(int)] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, 1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 1, 0])
            elif each_mesh == 9:  # up door
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1].astype(int)] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, 1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 1, 0])

            elif each_mesh == 4:  # handle
                links.append(part_path + part_names[each_mesh] + ".obj")
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_FIXED)
                link_scales.append([1, 1, 1])

                if each_parent - 1 >= len(mesh_pred):
                    link_positions.append([0, 0, 0])
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append([0, 0, 0, 1])
                    continue

                if mesh_pred[each_parent - 1] == 2:  # left door
                    link_positions.append(
                        [
                            0,
                            0.25
                            * scale_pred[each_parent - 1][1]
                            * position_type[position_pred[i][1].astype(int)],
                            0.25
                            * scale_pred[each_parent - 1][2]
                            * position_type[position_pred[i][2].astype(int)],
                        ]
                    )
                    linkJointAxis.append([1, 0, 0])
                    link_orientations.append([0, 0, 0, 1])
                elif mesh_pred[each_parent - 1] == 3:  # right door
                    link_positions.append(
                        [
                            0,
                            -0.25
                            * scale_pred[each_parent - 1][1]
                            * position_type[position_pred[i][1].astype(int)],
                            0.25
                            * scale_pred[each_parent - 1][2]
                            * position_type[position_pred[i][2].astype(int)],
                        ]
                    )
                    linkJointAxis.append([1, 0, 0])
                    link_orientations.append([0, 0, 0, 1])
                elif mesh_pred[each_parent - 1] == 7:
                    link_positions.append(
                        [
                            0,
                            0.25
                            * scale_pred[each_parent - 1][1]
                            * position_type[position_pred[i][1].astype(int)],
                            0.25
                            * scale_pred[each_parent - 1][2]
                            * position_type[position_pred[i][2].astype(int)],
                        ]
                    )
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append(
                        p.getQuaternionFromEuler([np.pi / 2, 0, 0])
                    )
                elif mesh_pred[each_parent - 1] == 8:
                    link_positions.append(
                        [
                            0,
                            0.25 * scale_pred[each_parent - 1][1] * position_type[6],
                            0.25 * scale_pred[each_parent - 1][2] * position_type[10],
                        ]
                    )
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append(
                        p.getQuaternionFromEuler([np.pi / 2, 0, 0])
                    )
                elif mesh_pred[each_parent - 1] == 9:
                    link_positions.append(
                        [
                            0,
                            0.25
                            * scale_pred[each_parent - 1][1]
                            * position_type[position_pred[i][1].astype(int)],
                            -0.25
                            * scale_pred[each_parent - 1][2]
                            * position_type[position_pred[i][2].astype(int)],
                        ]
                    )
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append(
                        p.getQuaternionFromEuler([np.pi / 2, 0, 0])
                    )
                elif mesh_pred[each_parent - 1] == 1:  # drawer
                    link_positions.append(
                        [
                            0,
                            0.25
                            * scale_pred[each_parent - 1][1]
                            * position_type[position_pred[i][1].astype(int)],
                            0.25
                            * scale_pred[each_parent - 1][2]
                            * position_type[position_pred[i][2].astype(int)],
                        ]
                    )
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append(
                        p.getQuaternionFromEuler([np.pi / 2, 0, 0])
                    )
                else:
                    link_positions.append([0, 0, 0])
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append([0, 0, 0, 1])

            elif each_mesh == 5:  # knob
                links.append(part_path + part_names[each_mesh] + ".obj")
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_FIXED)
                link_scales.append([1, 1, 1])
                if each_parent - 1 >= len(mesh_pred):
                    link_positions.append([0, 0, 0])
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append([0, 0, 0, 1])
                    continue
                if mesh_pred[each_parent - 1] == 2:  # left door
                    link_positions.append(
                        [
                            0,
                            0.25
                            * scale_pred[each_parent - 1][1]
                            * position_type[position_pred[i][1].astype(int)],
                            0.25
                            * scale_pred[each_parent - 1][2]
                            * position_type[position_pred[i][2].astype(int)],
                        ]
                    )
                    linkJointAxis.append([1, 0, 0])
                    link_orientations.append([0, 0, 0, 1])
                elif mesh_pred[each_parent - 1] == 3:  # right door
                    link_positions.append(
                        [
                            0,
                            -0.25
                            * scale_pred[each_parent - 1][1]
                            * position_type[position_pred[i][1].astype(int)],
                            0.25
                            * scale_pred[each_parent - 1][2]
                            * position_type[position_pred[i][2].astype(int)],
                        ]
                    )
                    linkJointAxis.append([1, 0, 0])
                    link_orientations.append([0, 0, 0, 1])
                elif mesh_pred[each_parent - 1] == 1:  # drawer
                    link_positions.append(
                        [
                            0,
                            0.25
                            * scale_pred[each_parent - 1][1]
                            * position_type[position_pred[i][1].astype(int)],
                            0.25
                            * scale_pred[each_parent - 1][2]
                            * position_type[position_pred[i][2].astype(int)],
                        ]
                    )
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append(
                        p.getQuaternionFromEuler([np.pi / 2, 0, 0])
                    )
                else:
                    link_positions.append([0, 0, 0])
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append([0, 0, 0, 1])

            elif each_mesh == 6:  # washer door
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1].astype(int)] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, -1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 0, 1])

            elif each_mesh == 7:  # down door
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1]] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, -1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 0, 1])

        abs_link_scales = [[abs(x) for x in sublist] for sublist in link_scales]


        obj = create_articulated_objects(
            root,
            root_scale,
            root_position,
            root_orientation,
            links,
            abs_link_scales,
            link_positions,
            link_orientations,
            linkparents,
            jointtypes,
            linkJointAxis,
            texture_list,
            if_random,
        )

    write_urdfs(
        filename,
        root,
        root_scale,
        root_position,
        root_orientation,
        links,
        link_scales,
        link_positions,
        link_orientations,
        linkparents,
        jointtypes,
        linkJointAxis,
    )

    return obj, link_orientations, tree

def visualization_parts(
    p,
    root_position,
    root_orientation,
    root_scale,
    mesh_base,
    position_pred_ori,
    scale_pred_ori,
    mesh_pred_ori,
    parent_pred_ori,
    texture_list,
    if_random,
    filename="output",
):
    num_parts = len(mesh_pred_ori)
    num_roots = 1

    part_names = [
        "none",
        "drawer",
        "doorL",
        "doorR",
        "handle",
        "knob",
        "washer_door",
        "doorD",
        "oven_door",
        "doorU",
    ]
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
    part_path = "meshes/parts/"

    # fix the oven fan problem: if there are parts, root shouldn't be oven fan
    if len(position_pred_ori) > 0 and mesh_base == 6:
        mesh_base = 1

    root = "meshes/{}.obj".format(base_names[mesh_base])

    position_type = np.arange(13) / 12
    links = []
    link_scales = []
    link_positions = []
    link_orientations = []
    linkparents = []
    jointtypes = []
    linkJointAxis = []

    parent_pred = []
    relations_pred = []

    if mesh_base >= 9:  # rigid objects
        object_path = "meshes/{}.obj".format(base_names[mesh_base])
        obj = create_obj(p, object_path, root_scale, root_position, root_orientation)
        p.changeVisualShape(obj, -1, rgbaColor=[0.6, 0.6, 0.6, 1])
        # if base_names[mesh_base]=="square_table":
        if mesh_base > 9:
            base_texture = random.choice(glob.glob("default_textures/cab_wood/*"))
            base_tex = p.loadTexture(base_texture)
            p.changeVisualShape(
                obj, -1, rgbaColor=(1, 1, 1, 1), textureUniqueId=base_tex
            )
    else:
        for i, each_parent in enumerate(parent_pred_ori[num_roots:]):
            each_parent = np.unravel_index(np.argmax(each_parent), each_parent.shape)
            parent_id = each_parent[0]
            # # [EDIT by ljy] if invalid parent prediction
            # if parent_id >= num_parts:
            #     parent_id = 0
            #########################################
            parent_pred.append(parent_id)
            relations_pred.append(each_parent[1])

        parent_pred = np.array(parent_pred)
        mesh_pred = np.array(mesh_pred_ori)
        scale_pred = np.array(scale_pred_ori)
        position_pred = np.array(position_pred_ori)

        # reorder the parts in ascending order according to the parent id
        new_order = np.argsort(parent_pred)  # ascending order
        parent_pred, scale_pred, mesh_pred, position_pred = (
            parent_pred[new_order],
            scale_pred[new_order],
            mesh_pred[new_order],
            position_pred[new_order],
        )
        if len(texture_list) > 0:
            texture_list = [texture_list[i] for i in new_order]

        # update parent labels
        for p_i, each_parent in enumerate(parent_pred):
            if each_parent > 0:
                if each_parent - 1 in list(new_order):
                    parent_pred[p_i] = list(new_order).index(each_parent - 1) + 1

        parents = {}
        children = {}
        # build the tree structure for visualization
        for i, each_parent in enumerate(parent_pred):
            parent_id = each_parent
            if i + 1 not in parents.keys():
                parents.update({i + 1: parent_id})
            if parent_id not in children.keys():
                children.update({parent_id: [i + 1]})
            else:
                children[parent_id].append(i + 1)
        
        tree = [{"id": 0, "children": [], "parent": -1, "name": "base"}]
        for i in range(len(parents)):
            tree.append({
                "id": i + 1,
                "children": [] if i + 1 not in children.keys() else children[i + 1],
                "parent": parents[i + 1],
            })

        for i, each_parent in enumerate(parent_pred):
            each_mesh = int(mesh_pred[i])
            parent_id = each_parent

            if each_mesh == 1:  # drawer
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append(
                    [1, scale_pred[i][1] - 0.02, scale_pred[i][2] - 0.02]
                )
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1].astype(int)] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, 1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_PRISMATIC)
                linkJointAxis.append([1, 0, 0])
                
            elif each_mesh == 2:  # left door
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append(
                    [1, scale_pred[i][1] - 0.01, scale_pred[i][2] - 0.01]
                )
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1].astype(int)] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, 1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 0, 1])
            elif each_mesh == 3:  # right door
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append(
                    [1, scale_pred[i][1] - 0.01, scale_pred[i][2] - 0.01]
                )
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1].astype(int)] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, -1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 0, 1])
            elif each_mesh == 6:  # washerdoor
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1].astype(int)] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, 1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 0, 1])
            elif each_mesh == 7:  # down door
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1].astype(int)] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, 1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 1, 0])
            elif each_mesh == 8:  # oven door:
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1].astype(int)] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, 1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 1, 0])
            elif each_mesh == 9:  # up door
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1].astype(int)] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, 1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 1, 0])

            elif each_mesh == 4:  # handle
                links.append(part_path + part_names[each_mesh] + ".obj")
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_FIXED)
                link_scales.append([1, 1, 1])

                if each_parent - 1 >= len(mesh_pred):
                    link_positions.append([0, 0, 0])
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append([0, 0, 0, 1])
                    continue

                if mesh_pred[each_parent - 1] == 2:  # left door
                    link_positions.append(
                        [
                            0,
                            0.25
                            * scale_pred[each_parent - 1][1]
                            * position_type[position_pred[i][1].astype(int)],
                            0.25
                            * scale_pred[each_parent - 1][2]
                            * position_type[position_pred[i][2].astype(int)],
                        ]
                    )
                    linkJointAxis.append([1, 0, 0])
                    link_orientations.append([0, 0, 0, 1])
                elif mesh_pred[each_parent - 1] == 3:  # right door
                    link_positions.append(
                        [
                            0,
                            -0.25
                            * scale_pred[each_parent - 1][1]
                            * position_type[position_pred[i][1].astype(int)],
                            0.25
                            * scale_pred[each_parent - 1][2]
                            * position_type[position_pred[i][2].astype(int)],
                        ]
                    )
                    linkJointAxis.append([1, 0, 0])
                    link_orientations.append([0, 0, 0, 1])
                elif mesh_pred[each_parent - 1] == 7:
                    link_positions.append(
                        [
                            0,
                            0.25
                            * scale_pred[each_parent - 1][1]
                            * position_type[position_pred[i][1].astype(int)],
                            0.25
                            * scale_pred[each_parent - 1][2]
                            * position_type[position_pred[i][2].astype(int)],
                        ]
                    )
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append(
                        p.getQuaternionFromEuler([np.pi / 2, 0, 0])
                    )
                elif mesh_pred[each_parent - 1] == 8:
                    link_positions.append(
                        [
                            0,
                            0.25 * scale_pred[each_parent - 1][1] * position_type[6],
                            0.25 * scale_pred[each_parent - 1][2] * position_type[10],
                        ]
                    )
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append(
                        p.getQuaternionFromEuler([np.pi / 2, 0, 0])
                    )
                elif mesh_pred[each_parent - 1] == 9:
                    link_positions.append(
                        [
                            0,
                            0.25
                            * scale_pred[each_parent - 1][1]
                            * position_type[position_pred[i][1].astype(int)],
                            -0.25
                            * scale_pred[each_parent - 1][2]
                            * position_type[position_pred[i][2].astype(int)],
                        ]
                    )
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append(
                        p.getQuaternionFromEuler([np.pi / 2, 0, 0])
                    )
                elif mesh_pred[each_parent - 1] == 1:  # drawer
                    link_positions.append(
                        [
                            0,
                            0.25
                            * scale_pred[each_parent - 1][1]
                            * position_type[position_pred[i][1].astype(int)],
                            0.25
                            * scale_pred[each_parent - 1][2]
                            * position_type[position_pred[i][2].astype(int)],
                        ]
                    )
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append(
                        p.getQuaternionFromEuler([np.pi / 2, 0, 0])
                    )
                else:
                    link_positions.append([0, 0, 0])
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append([0, 0, 0, 1])

            elif each_mesh == 5:  # knob
                links.append(part_path + part_names[each_mesh] + ".obj")
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_FIXED)
                link_scales.append([1, 1, 1])
                if each_parent - 1 >= len(mesh_pred):
                    link_positions.append([0, 0, 0])
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append([0, 0, 0, 1])
                    continue
                if mesh_pred[each_parent - 1] == 2:  # left door
                    link_positions.append(
                        [
                            0,
                            0.25
                            * scale_pred[each_parent - 1][1]
                            * position_type[position_pred[i][1].astype(int)],
                            0.25
                            * scale_pred[each_parent - 1][2]
                            * position_type[position_pred[i][2].astype(int)],
                        ]
                    )
                    linkJointAxis.append([1, 0, 0])
                    link_orientations.append([0, 0, 0, 1])
                elif mesh_pred[each_parent - 1] == 3:  # right door
                    link_positions.append(
                        [
                            0,
                            -0.25
                            * scale_pred[each_parent - 1][1]
                            * position_type[position_pred[i][1].astype(int)],
                            0.25
                            * scale_pred[each_parent - 1][2]
                            * position_type[position_pred[i][2].astype(int)],
                        ]
                    )
                    linkJointAxis.append([1, 0, 0])
                    link_orientations.append([0, 0, 0, 1])
                elif mesh_pred[each_parent - 1] == 1:  # drawer
                    link_positions.append(
                        [
                            0,
                            0.25
                            * scale_pred[each_parent - 1][1]
                            * position_type[position_pred[i][1].astype(int)],
                            0.25
                            * scale_pred[each_parent - 1][2]
                            * position_type[position_pred[i][2].astype(int)],
                        ]
                    )
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append(
                        p.getQuaternionFromEuler([np.pi / 2, 0, 0])
                    )
                else:
                    link_positions.append([0, 0, 0])
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append([0, 0, 0, 1])

            elif each_mesh == 6:  # washer door
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1].astype(int)] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, -1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 0, 1])

            elif each_mesh == 7:  # down door
                links.append(part_path + part_names[each_mesh] + ".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append(
                    [
                        0,
                        position_type[position_pred[i][1]] * root_scale[1],
                        position_type[position_pred[i][2].astype(int)] * root_scale[2],
                    ]
                )
                link_orientations.append([0, 0, 0, -1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 0, 1])

        abs_link_scales = [[abs(x) for x in sublist] for sublist in link_scales]


        obj = create_articulated_objects(
            root,
            root_scale,
            root_position,
            root_orientation,
            links,
            abs_link_scales,
            link_positions,
            link_orientations,
            linkparents,
            jointtypes,
            linkJointAxis,
            texture_list,
            if_random,
        )

    write_urdfs(
        filename,
        root,
        root_scale,
        root_position,
        root_orientation,
        links,
        link_scales,
        link_positions,
        link_orientations,
        linkparents,
        jointtypes,
        linkJointAxis,
    )

    return obj, link_orientations, tree


def write_numpy(
    filename,
    root,
    root_scale,
    root_position,
    root_orientation,
    links,
    link_scales,
    link_positions,
    link_orientations,
    linkparents,
    jointtypes,
    linkJointAxis,
):

    urdf_primitives = {}
    urdf_primitives["root"] = root
    urdf_primitives["root_scale"] = root_scale
    urdf_primitives["root_position"] = root_position
    urdf_primitives["root_orientation"] = root_orientation
    urdf_primitives["links"] = links
    urdf_primitives["link_scales"] = link_scales
    urdf_primitives["link_positions"] = link_positions
    urdf_primitives["link_orientations"] = link_orientations
    urdf_primitives["linkparents"] = linkparents
    urdf_primitives["jointtypes"] = jointtypes
    urdf_primitives["linkJointAxis"] = linkJointAxis

    np.save(filename, urdf_primitives)


def write_urdfs(
    filename,
    root,
    root_scale,
    root_position,
    root_orientation,
    links,
    link_scales,
    link_positions,
    link_orientations,
    linkparents,
    jointtypes,
    linkJointAxis,
):
    root_rot = Rot.from_quat(root_orientation).as_rotvec()
    import xml.etree.ElementTree as ET
    import os
    from xml.dom import minidom

    def prettify(elem):
        """Return a pretty-printed XML string for the Element."""
        rough_string = ET.tostring(elem, "utf-8")
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    # Create the root element of the URDF file, which is <robot>
    joint_names = ["revolute", "prismatic", "spherical", "unknown", "fixed"]
    robot = ET.Element("robot", attrib={"name": "my_urdf"})

    # Add a base link element with a mesh file for the visual geometry
    base_link = ET.SubElement(robot, "link", attrib={"name": "base_link"})
    visual = ET.SubElement(base_link, "visual")
    ET.SubElement(
        visual,
        "origin",
        attrib={
            "xyz": "{0} {1} {2}".format(
                root_position[0], root_position[1], root_position[2]
            ),
            "rpy": "{0} {1} {2}".format(root_rot[0], root_rot[1], root_rot[2]),
        },
    )

    material = ET.SubElement(visual, "material", attrib={"name": "white"})
    ET.SubElement(
        material, "color", attrib={"rgba": "{0} {1} {2} 1".format(0.8, 0.8, 0.8)}
    )

    visual_geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(
        visual_geometry,
        "mesh",
        attrib={
            "filename": "../meshes/cabinet.obj",
            "scale": "{0} {1} {2}".format(root_scale[0], root_scale[1], root_scale[2]),
        },
    )

    collision = ET.SubElement(base_link, "collision")
    ET.SubElement(
        collision,
        "origin",
        attrib={
            "xyz": "{0} {1} {2}".format(
                root_position[0], root_position[1], root_position[2]
            ),
            "rpy": "{0} {1} {2}".format(root_rot[0], root_rot[1], root_rot[2]),
        },
    )
    collision_geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(
        collision_geometry,
        "mesh",
        attrib={
            "filename": "../meshes/cabinet.obj",
            "scale": "{0} {1} {2}".format(root_scale[0], root_scale[1], root_scale[2]),
        },
    )

    inertial = ET.SubElement(base_link, "inertial")
    ET.SubElement(inertial, "mass", attrib={"value": "1"})
    ET.SubElement(
        inertial,
        "inertia",
        attrib={
            "ixx": "1e-4",
            "ixy": "0",
            "ixz": "0",
            "iyy": "1e-4",
            "iyz": "0",
            "izz": "1e-4",
        },
    )

    # Loop over links and create URDF elements for each
    link_names = ["base_link"]
    for link_id, link_info in enumerate(links):
        link_names.append(os.path.basename(link_info)[:-4] + "{}".format(link_id))

    for link_id, link_info in enumerate(links):
        link_rot = Rot.from_quat(link_orientations[link_id]).as_rotvec()
        if "drawer" in os.path.basename(link_info)[:-4]:
            limitL = 0
            limitU = 0.4
        elif "doorR" in os.path.basename(link_info)[:-4]:
            limitL = 0
            limitU = 1.57
        elif "doorL" in os.path.basename(link_info)[:-4]:
            limitL = -1.57
            limitU = 0
        elif "doorD" in os.path.basename(link_info)[:-4]:
            limitL = -1.57
            limitU = 0
        elif "doorU" in os.path.basename(link_info)[:-4]:
            limitL = 0
            limitU = 1.57
        else:
            limitL = 0
            limitU = 1.57

        # Add a link element for each link
        # link_names.append(os.path.basename(link_info)[:-4]+"{}".format(link_id))
        link = ET.SubElement(
            robot,
            "link",
            attrib={"name": os.path.basename(link_info)[:-4] + "{}".format(link_id)},
        )
        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
        material = ET.SubElement(visual, "material", attrib={"name": "white"})
        ET.SubElement(
            material, "color", attrib={"rgba": "{0} {1} {2} 1".format(0.8, 0.8, 0.8)}
        )
        visual_geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(
            visual_geometry,
            "mesh",
            attrib={
                "filename": "../meshes/parts/{}".format(os.path.basename(link_info)),
                "scale": "{0} {1} {2}".format(
                    link_scales[link_id][0],
                    link_scales[link_id][1],
                    link_scales[link_id][2],
                ),
            },
        )

        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
        collision_geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(
            collision_geometry,
            "mesh",
            attrib={
                "filename": "../meshes/parts/{}".format(os.path.basename(link_info)),
                "scale": "{0} {1} {2}".format(
                    link_scales[link_id][0],
                    link_scales[link_id][1],
                    link_scales[link_id][2],
                ),
            },
        )

        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", attrib={"value": "0.2"})
        ET.SubElement(
            inertial,
            "inertia",
            attrib={
                "ixx": "1e-4",
                "ixy": "0",
                "ixz": "0",
                "iyy": "1e-4",
                "iyz": "0",
                "izz": "1e-4",
            },
        )

        # Add a joint element for each link
        joint = ET.SubElement(
            robot,
            "joint",
            attrib={
                "name": "{0}_to_{1}".format(
                    link_names[link_id + 1], link_names[linkparents[link_id]]
                ),
                "type": joint_names[jointtypes[link_id]],
            },
        )
        ET.SubElement(
            joint,
            "axis",
            attrib={
                "xyz": "{0} {1} {2}".format(
                    linkJointAxis[link_id][0],
                    linkJointAxis[link_id][1],
                    linkJointAxis[link_id][2],
                )
            },
        )
        ET.SubElement(
            joint,
            "limit",
            attrib={
                "effort": "5",
                "lower": "{0}".format(limitL),
                "upper": "{0}".format(limitU),
                "velocity": "2.283",
            },
        )
        ET.SubElement(
            joint,
            "origin",
            attrib={
                "xyz": "{0} {1} {2}".format(
                    link_positions[link_id][0],
                    link_positions[link_id][1],
                    link_positions[link_id][2],
                ),
                "rpy": "{0} {1} {2}".format(link_rot[0], link_rot[1], link_rot[2]),
            },
        )

        ET.SubElement(
            joint,
            "parent",
            attrib={"link": "{}".format(link_names[linkparents[link_id]])},
        )
        ET.SubElement(
            joint, "child", attrib={"link": "{}".format(link_names[link_id + 1])}
        )

    # Once all elements are added, write the URDF to a file
    tree = ET.ElementTree(robot)

    robot_xml_str = prettify(robot)
    with open("{}.urdf".format(filename), "w") as file:
        file.write(robot_xml_str)

    tree.write("{}.urdf".format(filename), encoding="utf-8", xml_declaration=True)


def create_obj(p, obj_path, scale, obj_t, obj_q):
    base_visualid = p.createVisualShape(
        shapeType=p.GEOM_MESH, fileName=obj_path, rgbaColor=None, meshScale=list(scale)
    )

    base_collisionid = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=obj_path,
        meshScale=list(scale),
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
    )

    new_obj = p.createMultiBody(0, base_collisionid, base_visualid, obj_t, obj_q)

    return new_obj
