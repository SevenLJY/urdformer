# reference of object categories
# cat_ref = {
#     "Table": 0,
#     "Dishwasher": 1,
#     "StorageFurniture": 2,
#     "Refrigerator": 3,
#     "WashingMachine": 4,
#     "Microwave": 5,
#     "Oven": 6,
#     "Safe": 7,
# }

cat_ref = {
    "Remote": 0,
    "Switch": 1,
    "Fan": 2,
    "Faucet": 3,
    "Bottle": 4,
    "Eyeglasses": 5,
    "Laptop": 6,
    "Table": 7,
    "Chair": 8,
    "Dishwasher": 9,
    "Door": 10,
    "StorageFurniture": 11,
    "Display": 12,
    "USB": 13,
    "Globe": 14,
    "Pliers": 15,
    "Keyboard": 16,
    "Cart": 17,
    "Suitcase": 18,
    "Knife": 19,
    "Pen": 20,
    "Camera": 21,
    "Lighter": 22,
    "Lamp": 23,
    "Refrigerator": 24,
    "Clock": 25,
    "WashingMachine": 26,
    "Phone": 27,
    "Bucket": 28,
    "CoffeeMachine": 29,
    "Mouse": 30,
    "Window": 31,
    "Box": 32,
    "Microwave": 33,
    "Toilet": 34,
    "Toaster": 35,
    "TrashCan": 36,
    "Printer": 37,
    "Stapler": 38,
    "Kettle": 39,
    "Scissors": 40,
    "Oven": 41,
    "Safe": 42,
    "FoldingChair": 43,
    "Dispenser": 44,
    "KitchenPot": 45
}

# reference of semantic labels for each part
sem_ref = {
    "fwd": {
        "door": 0,
        "drawer": 1,
        "base": 2,
        "handle": 3,
        "wheel": 4,
        "knob": 5,
        "shelf": 6,
        "tray": 7
    },
    "bwd": {
        0: "door",
        1: "drawer",
        2: "base",
        3: "handle",
        4: "wheel",
        5: "knob",
        6: "shelf",
        7: "tray"
    }
}

# reference of joint types for each part
joint_ref = {
    "fwd": {
       "fixed": 1,
        "revolute": 2,
        "prismatic": 3,
        "screw": 4,
        "continuous": 5 
    },
    "bwd": {
        1: "fixed",
        2: "revolute",
        3: "prismatic",
        4: "screw",
        5: "continuous"
    } 
}



import plotly.express as px
# pallette for joint type color
joint_color_ref = px.colors.qualitative.Set1
# pallette for graph node color
graph_color_ref = px.colors.qualitative.Bold + px.colors.qualitative.Prism
# pallette for semantic label color
semantic_color_ref = px.colors.qualitative.Vivid_r
# attention map visulaization color
attn_color_ref = px.colors.sequential.Viridis

from matplotlib.colors import LinearSegmentedColormap
cmap_attn = LinearSegmentedColormap.from_list("mycmap", attn_color_ref, N=256)