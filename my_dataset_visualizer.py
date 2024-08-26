import json
import os

import cv2
import numpy as np
import torch
from my_dataset import PMDataset
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

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

RG2CODE = {"none": 0,
           "cabinet_kitchen": 1,
           "oven": 2,
           "dishwasher": 3,
           "washer": 4,
           "fridge": 5,
           "table": 6,
           "microwave": 7}

CODE2RG = {0: "none",
           1: "cabinet_kitchen",
           2: "oven",
           3: "dishwasher",
           4: "washer",
           5: "fridge",
           6: "table",
           7: "microwave"}


class DatasetVisualizer:
    def __init__(self, dataset, output_dir):
        self.dataset = dataset
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def visualize(self, num_samples=10):
        html_content = """
        <html>
        <head>
            <style>
                table, th, td {
                    border: 1px solid black;
                    border-collapse: collapse;
                    padding: 5px;
                }
                img {
                    max-width: 300px;
                }
            </style>
        </head>
        <body>
            <table>
                <tr>
                    <th>Model ID + View ID</th>
                    <th>Image with Bounding Boxes</th>
                    <th>Additional Information</th>
                </tr>
        """

        for i in range(len(self.dataset)):
            input_data, supervision = self.dataset[i]
            model_id = self.dataset.model_ids[self.dataset.index_map[i]].split("/")[-1]
            view_id = i % self.dataset.cfg.dataset.n_views

            # Generate image with bounding boxes
            img_with_bbox = self._draw_bounding_boxes(input_data)
            img_filename = f"{model_id}_{view_id:02d}.png"
            img_path = os.path.join(self.output_dir, img_filename)
            img_with_bbox.save(img_path)

            # Generate additional information
            additional_info = self._generate_additional_info(supervision)

            html_content += f"""
                <tr>
                    <td>{model_id} + {view_id:02d}</td>
                    <td><img src="{img_filename}"></td>
                    <td>{additional_info}</td>
                </tr>
            """

        html_content += """
            </table>
        </body>
        </html>
        """

        with open(os.path.join(self.output_dir, "visualization.html"), "w") as f:
            f.write(html_content)

    def _draw_bounding_boxes(self, input_data):
        # Denormalize and convert the image tensor to PIL Image
        img_tensor = input_data["img"]
        img_denorm = self._denormalize_image(img_tensor)
        img_pil = transforms.ToPILImage()(img_denorm)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.load_default()

        # Draw bounding boxes
        bboxes = input_data["bbox"][0]
        for i, bbox in enumerate(bboxes):
            if np.all(bbox == 0):
                continue
            y, x, h, w = bbox
            x1, y1 = x * 224, y * 224
            x2, y2 = (x + w) * 224, (y + h) * 224
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1 + 3, y1 + 3), str(i + 1), font=font, fill="blue")

        return img_pil

    def _denormalize_image(self, img_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return img_tensor * std + mean

    def _generate_additional_info(self, supervision):
        base_type = supervision["base_type"].item()
        mesh_types = supervision["mesh_types"]
        positions_min, positions_max = supervision["positions"]
        connectivity = supervision["connectivity"]

        info = f"Base Type: {base_type} ({CODE2RG[base_type]})<br>"
        info += "Part Information:<br>"
        for i, mesh_type in enumerate(mesh_types):
            if mesh_type == 0:
                continue
            parent_id = np.where(connectivity[i+1, :, 0] == 1)[0][0]
            info += f"ID {i + 1}: Mesh Type {mesh_type} ({part_names[mesh_type]}), Parent ID {parent_id}, "
            info += f"Position Start {positions_min[i]}, Position End {positions_max[i]}<br>"

        return info


# Usage
cfg = "./my_cfg.yaml"
cfg = OmegaConf.load(cfg)
cfg.dataset.n_views = 1
dataset = PMDataset(cfg, split="test")
visualizer = DatasetVisualizer(dataset, output_dir="dataset_visualization")
visualizer.visualize(num_samples=200)
