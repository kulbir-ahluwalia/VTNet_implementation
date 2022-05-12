import time
import torch
import copy

from agents.navigation_agent import NavigationAgent
from data.constants import AI2THOR_TARGET_CLASSES
from models.model_io import ModelOptions
from models.vtnet import VTNet

from .train_util import (
    new_episode,
    run_episode,
    end_episode,
    reset_player,
    compute_spl,
    get_bucketed_metrics,
)


def a3c_val(
    args,
    saved_model,
    res_queue,
    max_count,
    scene_type,
    scenes,
    device,
):

    targets = AI2THOR_TARGET_CLASSES[22]

    if scene_type == "living_room":
        args.max_episode_length = 200
    else:
        args.max_episode_length = 100

    saved_state = torch.load(saved_model, map_location=device)
    args.use_nn_transformer = saved_state["args"].use_nn_transformer
    model = VTNet(device=device, use_nn_transformer=args.use_nn_transformer)
    model.load_state_dict(saved_state["model"])

    player = NavigationAgent(args, scenes, targets, device)
    player.sync_with_shared(model)
    count = 0

    model_options = ModelOptions()

    while count < max_count:
        # Get a new episode.
        total_reward = 0
        player.eps_len = 0
        new_episode(args, player)
        player_start_state = copy.deepcopy(player.environment.controller.state)
        player_start_time = time.time()

        while not player.done:
            # Make sure model is up to date.
            player.sync_with_shared(model)
            # Run episode for num_steps or until player is done.
            total_reward = run_episode(player, args, total_reward, model_options, False, model)
            # Compute the loss.
            # loss = compute_loss(args, player, gpu_id, model_options)
            if not player.done:
                reset_player(player)

        # for k in loss:
        #     loss[k] = loss[k].item()
        spl, best_path_length = compute_spl(player, player_start_state)

        bucketed_spl = get_bucketed_metrics(spl, best_path_length, player.success)

        end_episode(
            player,
            res_queue,
            total_time=time.time() - player_start_time,
            total_reward=total_reward,
            spl=spl,
            **bucketed_spl,
        )

        count += 1
        reset_player(player)

    player.exit()
    res_queue.put({"END": True})
