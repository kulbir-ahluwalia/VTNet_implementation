""" Base class for all Agents. """
from __future__ import division

import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from data.constants import DONE_ACTION_INT, AI2THOR_TARGET_CLASSES


class ThorAgent:
    """ Base class for all actor-critic agents. """

    def __init__(
        self, model, args, scenes, targets, device, episode=None, max_episode_length=1e3,
    ):
        self.scenes = scenes
        self.targets = targets
        self.targets_index = [i for i, item in enumerate(AI2THOR_TARGET_CLASSES[22]) if item in self.targets]

        self.device = device

        self._model = None
        self.model = model
        self._episode = episode
        self.eps_len = 0
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = False
        self.info = None
        self.reward = 0
        self.max_length = False
        self.hidden = None
        self.actions = []
        self.probs = []
        self.last_action_probs = None
        self.memory = []
        self.done_action_probs = []
        self.done_action_targets = []
        self.max_episode_length = max_episode_length
        self.success = False
        self.backprop_t = 0

        self.verbose = args.verbose
        self.learned_loss = False
        self.learned_input = None
        self.learned_t = 0
        self.num_steps = 50 
        self.hidden_state_sz = 512 
        self.action_space = 6

        self.targets_types = None
        # self.model_name = args.model

        self.action_num = 0
        self.meta_learning_actions = {}
        self.meta_predictions = []
        self.meta_duplicate_action = False
        self.meta_failed_action = False

        self.memory_duplicate_learning = False
        self.duplicate_states_actions = {}

        # imitation learning related parameters
        self.imitation_learning = False
        self.il_duplicate_action = False
        self.il_failed_action = False
        self.il_update_actions = {}

        self.record_attention = False # args.record_attention

    def sync_with_shared(self, shared_model):
        """ Sync with the shared model. """
        self.model.load_state_dict(shared_model.state_dict())

    def eval_at_state(self, model_options):
        """ Eval at state. """
        raise NotImplementedError()

    @property
    def episode(self):
        """ Return the current episode. """
        return self._episode

    @property
    def environment(self):
        """ Return the current environmnet. """
        return self.episode.environment

    @property
    def state(self):
        """ Return the state of the agent. """
        raise NotImplementedError()

    @state.setter
    def state(self, value):
        raise NotImplementedError()

    @property
    def model(self):
        """ Returns the model. """
        return self._model

    def print_info(self):
        """ Print the actions. """
        for action in self.actions:
            print(action)

    @model.setter
    def model(self, model_to_set):
        self._model = model_to_set
        if self._model is not None:
            self._model = self.model.to(self.device)

    def _increment_episode_length(self):
        self.eps_len += 1
        if self.eps_len >= self.max_episode_length:
            if not self.done:
                self.max_length = True
                self.done = True
            else:
                self.max_length = False
        else:
            self.max_length = False

    def action(self, model_options, training, test_update):
        """ Train the agent. """
        if training or test_update:
            self.model.train()
        else:
            self.model.eval()

        self.episode.states.append(str(self.episode.environment.controller.state))

        model_input, out = self.eval_at_state(model_options)

        # record the output of the model
        self.hidden = out.hidden

        # if out.state_representation is not None:
        #     self.episode.state_reps.append(out.state_representation.squeeze().cpu())
        # if out.state_memory is not None:
        #     self.episode.state_memory.append(out.state_memory.squeeze().cpu())
        # if out.action_memory is not None:
        #     self.episode.action_memory.append(out.action_memory.squeeze().cpu())
        # if out.obs_rep is not None:
        #     self.episode.obs_reps.append(out.obs_rep.squeeze().cpu())

        # if out.meta_action is not None:
        #     self.meta_predictions.append(F.softmax(out.meta_action, dim=1))
        #     self.episode.meta_predictions.append(F.softmax(out.meta_action, dim=1))

        # agent operates the asked action
        prob = F.softmax(out.logit, dim=1)
        # prob = F.softmax(out.meta_action, dim=1)
        self.episode.action_outputs.append(prob.tolist())
        if training:
            action = prob.multinomial(1).data
        else:
            action = prob.argmax(dim=1, keepdim=True)


        # act_action = action

        # num_ = min(2, len(self.episode.states))
        # if str(self.episode.environment.controller.state) in self.episode.states[:-num_]:
        #     action = F.softmax(out.meta_action + 0.3*out.logit, dim=1).argmax(dim=1, keepdim=True)
        #     # action = act_action + action





        log_prob = F.log_softmax(out.logit, dim=1)
        # log_prob = F.log_softmax(out.meta_action, dim=1)
        self.last_action_probs = prob
        entropy = -(log_prob * prob).sum(1)
        log_prob = log_prob.gather(1, Variable(action))

        self.reward, self.done, self.info = self.episode.step(action[0, 0])
        self.action_num += 1

        # record the actions those should be used to compute the loss
        meta_update_step = False
        # if self.episode.meta_learning:
        #     if self.meta_duplicate_action:
        #         if str(self.episode.environment.controller.state) in self.episode.states:
        #             meta_update_step = True
        #     elif self.meta_failed_action:
        #         if not self.info:
        #             meta_update_step = True

        if meta_update_step:
            optimal_action = self.environment.controller.get_optimal_action(self.episode.target_object)
            self.meta_learning_actions[self.action_num - 1] = optimal_action

        # record the actions those should be used to compute the loss during imitation learning
        imitation_learning_update = False
        if self.imitation_learning:
            if self.il_duplicate_action and str(self.episode.environment.controller.state) in self.episode.states:
                imitation_learning_update = True
            elif self.il_failed_action and not self.info:
                imitation_learning_update = True
            else:
                imitation_learning_update = True

        if imitation_learning_update:
            optimal_action = self.environment.controller.get_optimal_action(self.episode.target_object)
            self.il_update_actions[self.action_num-1] = optimal_action

        if self.memory_duplicate_learning and str(self.episode.environment.controller.state) in self.episode.states:
            optimal_action = self.environment.controller.get_optimal_action(self.episode.target_object)
            self.duplicate_states_actions[self.action_num-1] = optimal_action

        if self.verbose:
            print(self.episode.actions_list[action])
        self.probs.append(prob)
        self.episode.action_probs.append(prob)
        self.entropies.append(entropy)
        self.values.append(out.value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        self.actions.append(action)
        self.episode.actions_record.append(action)
        self.episode.prev_frame = model_input.state
        self.episode.current_frame = self.state()

        if self.learned_loss:
            res = torch.cat((self.hidden[0], self.last_action_probs), dim=1)
            if self.learned_input is None:
                self.learned_input = res
            else:
                self.learned_input = torch.cat((self.learned_input, res), dim=0)

        self._increment_episode_length()

        if self.episode.strict_done and action == DONE_ACTION_INT:
            self.success = self.info
            self.done = True
        elif self.done:
            self.success = not self.max_length

        return out.value, prob, action

    def reset_hidden(self, volatile=False):
        """ Reset the hidden state of the LSTM. """
        raise NotImplementedError()

    def repackage_hidden(self, volatile=False):
        """ Repackage the hidden state of the LSTM. """
        raise NotImplementedError()

    def clear_actions(self):
        """ Clear the information stored by the agent. """
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.probs = []
        self.reward = 0
        self.backprop_t = 0
        self.memory = []
        self.done_action_probs = []
        self.done_action_targets = []
        self.learned_input = None
        self.learned_t = 0
        self.il_update_actions = {}
        self.action_num = 0
        self.meta_learning_actions = {}
        self.meta_predictions = []
        self.duplicate_states_actions = {}
        return self

    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        raise NotImplementedError()

    def exit(self):
        """ Called on exit. """
        pass

    def reset_episode(self):
        """ Reset the episode so that it is identical. """
        return self._episode.reset()
