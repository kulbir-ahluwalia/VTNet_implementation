import time
from agents.navigation_agent import NavigationAgent
import torch

from data.data import num_to_name
from models.model_io import ModelOptions

from .train_util import (
    compute_loss,
    new_episode,
    run_episode,
    transfer_gradient_from_player_to_shared,
    end_episode,
    reset_player,
)


def a3c_train(
        args,
        model,
        optimizer,
        res_queue,
        end_flag,
        scenes,
        target_classes,
        device,
):
    targets = target_classes[22]

    player = NavigationAgent(args, scenes, targets, device)
    compute_grad = True

    model_options = ModelOptions()

    episode_num = 0

    while not end_flag.value:
        # Get a new episode.
        total_reward = 0
        player.eps_len = 0
        player.episode.episode_times = episode_num
        new_episode(args, player)
        player_start_time = time.time()

        # Train on the new episode.
        while not player.done:
            # Make sure model is up to date.
            player.sync_with_shared(model)
            # Run episode for num_steps or until player is done.
            total_reward = run_episode(player, args, total_reward, model_options, True)
            # Compute the loss.
            loss = compute_loss(args, player, device, model_options)
            if compute_grad and loss['total_loss'] != 0:
                # Compute gradient.
                player.model.zero_grad()
                loss['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(player.model.parameters(), 100.0)
                # Transfer gradient to shared model and step optimizer.
                transfer_gradient_from_player_to_shared(player, model)
                optimizer.step()
                # Clear actions and repackage hidden.
            if not player.done:
                reset_player(player)

        for k in loss:
            loss[k] = loss[k].item()

        end_episode(
            player,
            res_queue,
            title=num_to_name(int(player.episode.scene[9:])),
            total_time=time.time() - player_start_time,
            total_reward=total_reward,
        )
        reset_player(player)

        episode_num = (episode_num + 1) % len(args.scene_types)

    player.exit()
