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
        "individual": {
            "ID": {},
            "AID": {},
            "RID": {},
            "RIDD": {},
            "AOR": {},
        },
        "overall": {
            "ID": 0,
            "AID": 0,
            "RID": 0,
            "RIDD": 0,
            "AOR": 0,
        }
    }

    valid_aor_cnt = 0
    for case in tqdm(cases):
        tokens = case.split("_")
        model_id = f"{tokens[0]}/{tokens[1]}"
        pred = json.load(open(os.path.join(args.exp_dir, case, "object.json"), "r"))
        gt = json.load(open(os.path.join(args.gt_dir, model_id, "train_v3.json"), "r"))

        rid, ridd = RID(gt, pred, iou_include_base=args.include_base, use_gIoU=True)
        aid = AID(gt, pred, iou_include_base=args.include_base, use_gIoU=True, compare_handles=True)
        id = ID(pred, os.path.join(args.exp_dir, case), gt, os.path.join(args.gt_dir, model_id))
        aor = AOR(pred)

        rid = float(rid)
        ridd = float(ridd)
        aid = float(aid)
        id = float(id)
        aor = float(aor)

        metrics["individual"]["AID"][model_id] = aid
        metrics["individual"]["RID"][model_id] = rid
        metrics["individual"]["RIDD"][model_id] = ridd
        metrics["individual"]["ID"][model_id] = id

        if aor != -1:
            valid_aor_cnt += 1
            metrics["individual"]["AOR"][model_id] = aor
            metrics["overall"]["AOR"] += aor
        else:
            metrics["individual"]["AOR"][model_id] = None

        metrics["overall"]["AID"] += aid
        metrics["overall"]["RID"] += rid
        metrics["overall"]["RIDD"] += ridd
        metrics["overall"]["ID"] += id
    
    num_cases = len(cases)
    metrics["overall"]["AID"] /= num_cases
    metrics["overall"]["RID"] /= num_cases
    metrics["overall"]["RIDD"] /= num_cases
    metrics["overall"]["ID"] /= num_cases
    metrics["overall"]["AOR"] /= valid_aor_cnt

    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)
