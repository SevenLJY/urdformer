import os
import json
import argparse
from tqdm import tqdm
from my_metrics.aid import AID
from my_metrics.id import ID
from my_metrics.rid import RID
from my_metrics.aor import AOR

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
    cases.remove("collect.html")
    cases.sort()

    metrics = {
        "AID-IoU": 0,
        "RID-IoU": 0,
        "RID-cDist": 0,
        "AID-CD": 0,
        "RID-CD": 0,
        "AOR": 0,
    }

    valid_aor_cnt = 0
    for case in tqdm(cases):
        tokens = case.split("_")
        model_id = f"{tokens[0]}/{tokens[1]}"
        pred = json.load(open(os.path.join(args.exp_dir, case, "object.json"), "r"))
        gt = json.load(open(os.path.join(args.gt_dir, model_id, "train_v3.json"), "r"))

        rid_iou, rid_cdist = RID(gt, pred, iou_include_base=args.include_base, use_gIoU=True)
        aid_iou = AID(gt, pred, iou_include_base=args.include_base, use_gIoU=True, compare_handles=True)
        aid_cd, rid_cd = ID(pred, os.path.join(args.exp_dir, case), gt, os.path.join(args.gt_dir, model_id))
        aor = AOR(pred)

        rid_cdist = float(rid_cdist)
        rid_iou = float(rid_iou)
        aid_iou = float(aid_iou)
        aid_cd = float(aid_cd)
        rid_cd = float(rid_cd)
        aor = float(aor)

        metrics["AID-CD"] += aid_cd
        metrics["RID-CD"] += rid_cd
        metrics["AID-IoU"] += aid_iou
        metrics["RID-IoU"] += rid_iou
        metrics["RID-cDist"] += rid_cdist

        if aor != -1: # -1 means the model is invalid
            valid_aor_cnt += 1
            metrics["AOR"] += aor
        
    num_cases = len(cases)
    metrics["AID-CD"] /= num_cases
    metrics["RID-CD"] /= num_cases
    metrics["AID-IoU"] /= num_cases
    metrics["RID-IoU"] /= num_cases
    metrics["RID-cDist"] /= num_cases
    metrics["AOR"] /= valid_aor_cnt

    with open(f'{args.exp_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f)
