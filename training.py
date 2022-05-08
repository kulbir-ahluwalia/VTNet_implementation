import os
import argparse
from datetime import datetime
import ctypes
import time

import torch
import torch.multiprocessing as mp

from models.vtnet import VTNet
from globals import device
from data.constants import AI2THOR_TARGET_CLASSES
from runners.a3c_train import a3c_train
from utils.data_utils import loading_scene_list


def createOutputDirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)

    path = f"{path}/{datetime.now().strftime('%Y%m%d-%H%M%S%f')}"
    os.mkdir(path)
    return path


def main(args):
    outdir = createOutputDirectory(args.outdir)
    model = VTNet(device, args.use_nn_transformer)#.to(device)
    start_time_str = time.strftime(
        '%Y-%m-%d_%H-%M-%S', time.localtime(time.time())
    )
    train_total_ep = 0
    n_frames = 0
    args.num_steps = 50
    args.gamma = 0.99
    args.tau = 1.
    args.beta = 1e-2
    args.scene_types = ['kitchen', 'living_room', 'bedroom', 'bathroom']
    args.title = "a3c_vtnet"

    if args.pretrained_vtnet is not None:
        saved = torch.load(args.pretrained_vtnet, map_location=device)
        assert args.use_nn_transformer == saved["args"].use_nn_transformer
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in saved['model'].items() if (k in model_dict and v.shape == model_dict[k].shape)
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if args.pretrained_vtnet is not None:
        optimizer = torch.optim.Adam([
                {'params': [v for k, v in model
                    .named_parameters() if v.requires_grad and (k in pretrained_dict)],
                    'lr': args.pretrained_lr
                },
                {
                    'params': [v for k, v in model.named_parameters() if v.requires_grad and (k not in pretrained_dict)],
                    'lr': args.lr
                },
            ])
    else:
        optimizer = torch.optim.Adam(
            [v for k, v in model.named_parameters() if v.requires_grad], lr=args.lr
        )

    if args.init_model is not None:
        saved_state = torch.load(args.init_model, map_location=device)
        model.load_state_dict(saved_state["model"])
        optimizer.load_state_dict(saved_state["optimizer"])
        train_total_ep = saved_state["episodes"]
        n_frames = saved_state["frames"]

    target = a3c_train # if not args.eval else a3c_eval
    end_flag = mp.Value(ctypes.c_bool, False)
    train_res_queue = mp.Queue()
    scenes = loading_scene_list("train")
    processes = []

    for _ in range(0, args.workers):
        p = mp.Process(
            target=target,
            args=(
                args,
                model,
                optimizer,
                train_res_queue,
                end_flag,
                scenes,
                AI2THOR_TARGET_CLASSES,
                device,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)

    print("Train agents created.")
    start_time = time.time()

    try:
        while train_total_ep < args.max_ep:
            train_result = train_res_queue.get()
            train_total_ep += 1
            n_frames += train_result['ep_length']

            if (train_total_ep % args.save_every) == 0:
                print('{}: {}: {}'.format(
                    train_total_ep, n_frames, time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())))
                )
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'episodes': train_total_ep,
                    'frames': n_frames,
                    'args': args,
                }
                save_path = os.path.join(
                    outdir,
                    '{0}_{1}_{2}_{3}.dat'.format(
                        args.title, n_frames, train_total_ep, start_time_str
                    ),
                )
                torch.save(state, save_path)

            if train_total_ep % 1 == 0:
                print('{} s/ep'.format(time.time() - start_time))
                start_time = time.time()

    finally:
        end_flag.value = True
        for p in processes:
            time.sleep(0.1)
            p.join()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(description="VTNet training.")
    parser.add_argument("--data-dir", type=str, required=True, dest="data_dir", help="Data directory of training data")
    parser.add_argument("--out-dir", type=str, required=True, dest="outdir", help="Output directory")
    parser.add_argument("--lr", type=float, required=False, dest="lr", help="Learning rate", default=0.0001)
    parser.add_argument("--pretrained-lr", type=float, required=False, dest="pretrained_lr", help="Learning rate", default=0.00001)
    parser.add_argument("--workers", type=int, required=False, dest="workers", help="Number of workers", default=12)
    parser.add_argument("--max-ep", type=int, required=False, dest="max_ep", help="Number of epochs", default=60000)
    parser.add_argument("--save-every", type=int, required=False, dest="save_every", help="Save trained models after {save-every} epochs", default=1000)
    parser.add_argument("--use-nn-transformer", action="store_true", dest="use_nn_transformer", help="Use torch.nn.Transformer")
    parser.add_argument("--pretrained-vtnet", dest="pretrained_vtnet", required=False, help="Pretrained VTNet")
    parser.add_argument("--verbose", action="store_true", dest="verbose", help="Verbose output")
    parser.add_argument("--init-model", dest="init_model", required=False, help="Saved model")
    parser.add_argument("--num-workers", type=int, required=False, dest="num_workers", help="Number of workers", default=4)

    args = parser.parse_args()
    main(args)
