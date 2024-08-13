import os
import json
import argparse
import networkx as nx
from tqdm import tqdm
from my_metrics.cd import CD
from my_metrics.aor import AOR
from my_metrics.iou_cdist import IoU_cDist

def get_hash(file, dag=True):
    tree = file["diffuse_tree"]
    G = nx.DiGraph() if dag else nx.Graph()
    for node in tree:
        G.add_node(node["id"])
        if node["parent"] != -1:
            G.add_edge(node["id"], node["parent"])
    hashcode = nx.weisfeiler_lehman_graph_hash(G)
    return hashcode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="exps/freezed",
        help="The directory of the predicted results",
    )
    parser.add_argument(
        "--gt_dir", type=str, default="../data", help="The directory of the GT data"
    )
    parser.add_argument(
        "--include_base",
        action="store_true",
        help="Whether to include the base part when computing IoU",
    )

    args = parser.parse_args()

    cases = os.listdir(args.exp_dir)
    if os.path.exists(f'{args.exp_dir}/collect.html'):
        cases.remove("collect.html")
    if os.path.exists(f'{args.exp_dir}/metrics.json'):
        cases.remove("metrics.json")
    cases.sort()

    include_base = False

    metrics = {
        "AID-IoU": 0,
        "RID-IoU": 0,
        "AID-cDist": 0,
        "RID-cDist": 0,
        "AID-CD": 0,
        "RID-CD": 0,
        "AOR": 0,
        "graph_acc": 0
    }

    valid_aor_cnt = 0
    i = 0
    for case in tqdm(cases):
        tokens = case.split("_")
        model_id = f"{tokens[0]}/{tokens[1]}"
        pred = json.load(open(os.path.join(args.exp_dir, case, "object.json"), "r"))
        gt = json.load(open(os.path.join(args.gt_dir, model_id, "train_v3.json"), "r"))

        hash_pred = get_hash(pred)
        hash_gt = get_hash(gt)
        if hash_pred == hash_gt:
            metrics["graph_acc"] += 1

        scores = IoU_cDist(pred, gt, compare_handles=True, iou_include_base=include_base)
        cds = CD(pred, os.path.join(args.exp_dir, case), gt, os.path.join(args.gt_dir, model_id), include_base=include_base)
        aor = AOR(pred)

        aid_cdist = scores['AID-cDist']
        rid_cdist = scores['RID-cDist']
        aid_iou = scores['AID-IoU']
        rid_iou = scores['RID-IoU']
        aid_cd = cds['AID-CD']
        rid_cd = cds['RID-CD']


        metrics["AID-CD"] += aid_cd
        metrics["RID-CD"] += rid_cd
        metrics["AID-IoU"] += aid_iou
        metrics["RID-IoU"] += rid_iou
        metrics["AID-cDist"] += aid_cdist
        metrics["RID-cDist"] += rid_cdist


        if aor != -1: # -1 means the model is invalid for evaluating AOR
            valid_aor_cnt += 1
            metrics["AOR"] += aor
        i += 1
        
    num_cases = len(cases)
    metrics["AID-CD"] /= num_cases
    metrics["RID-CD"] /= num_cases
    metrics["AID-IoU"] /= num_cases
    metrics["RID-IoU"] /= num_cases
    metrics["AID-cDist"] /= num_cases
    metrics["RID-cDist"] /= num_cases
    metrics["AOR"] /= valid_aor_cnt
    metrics["graph_acc"] /= num_cases

    with open(f'{args.exp_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f)
