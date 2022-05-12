import os
import json
import argparse
import torch

from main_eval import main_eval
from globals import device
from tqdm import tqdm
from tabulate import tabulate


os.environ["OMP_NUM_THREADS"] = "1"


def full_eval(args):
    assert args is not None
    
    outdir = os.path.join(args.saved_model_dir, "validation")
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    args.phase = 'eval'
    args.episode_type = 'TestValEpisode'
    # args.test_or_val = 'val'

    args.results_json = os.path.join(outdir, 'result.json')

    # Get all valid saved_models for the given title and sort by train_ep.
    checkpoints = [(f, f.split("_")) for f in os.listdir(args.saved_model_dir)]
    checkpoints = [
        (f, int(s[-3]))
        for (f, s) in checkpoints
        if len(s) >= 4 # and f.startswith(args.title) and int(s[-3]) >= args.test_start_from
    ]
    checkpoints.sort(key=lambda x: x[1])

    best_model_on_val = None
    best_performance_on_val = 0.0

    # To find best model
    for (f, train_ep) in tqdm(checkpoints, desc="Checkpoints."):

        model = os.path.join(args.saved_model_dir, f)
        # args.load_model = model

        # run eval on model
        # args.test_or_val = "val"
        main_eval(args, saved_model=model, outdir=outdir, device=device)

        # check if best on val.
        with open(args.results_json, "r") as f:
            results = json.load(f)

        if results["success"] > best_performance_on_val:
            best_model_on_val = model
            best_performance_on_val = results["success"]

        print(f"{'=' * 20}\nModel: {model}")
        print(f"val/success: {results['success']}, {train_ep}")
        print(f"val/spl: {results['spl']}, {train_ep}")

    # Evaluate on the test dataset
    # args.test_or_val = "test"
    args.load_model = best_model_on_val
    main_eval(args, saved_model=best_model_on_val, outdir=outdir, device=device)

    with open(args.results_json, "r") as f:
        results = json.load(f)

    print(
        tabulate(
            [
                ["SPL >= 1:", results["GreaterThan/1/spl"]],
                ["Success >= 1:", results["GreaterThan/1/success"]],
                ["SPL >= 5:", results["GreaterThan/5/spl"]],
                ["Success >= 5:", results["GreaterThan/5/success"]],
            ],
            headers=["Metric", "Result"],
            tablefmt="orgtbl",
        )
    )

    print("Best model:", args.load_model)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(description="VTNet training.")
    parser.add_argument("--data-dir", type=str, required=True, dest="data_dir", help="Data directory of val/test data")
    parser.add_argument("--saved-model-dir", type=str, required=True, dest="saved_model_dir", help="Saved model directory")
    parser.add_argument("--num-workers", type=int, required=False, dest="num_workers", help="Number of workers", default=4)
    parser.add_argument("--verbose", action="store_true", dest="verbose", help="Verbose output")

    args = parser.parse_args()
    full_eval(args=args)
