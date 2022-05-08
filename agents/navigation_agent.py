from models.vtnet import VTNet
import torch
import numpy as np
import h5py
import os

from models.model_io import ModelInput
from episodes.basic_episode import BasicEpisode

from .agent import ThorAgent


class NavigationAgent(ThorAgent):
    """ A navigation agent who learns with pretrained embeddings. """

    def __init__(self, args, scenes, targets, device):
        max_episode_length = 30
        hidden_state_sz = 512
        self.action_space = 6
        self.device = device

        episode = BasicEpisode(args=args, device=device, strict_done=True)

        super(NavigationAgent, self).__init__(
            VTNet(device=device, use_nn_transformer=args.use_nn_transformer), args, scenes, targets, device, episode, max_episode_length,
        )
        self.hidden_state_sz = hidden_state_sz
        self.model_name = "VTNetModel"

        self.glove = {}
        # if 'SP' in self.model_name:
        #     with h5py.File(os.path.expanduser('~/Code/vn/glove_map300d.hdf5'), 'r') as rf:
        #         for i in rf:
        #             self.glove[i] = rf[i][:]


    def eval_at_state(self, model_options):
        model_input = ModelInput()

        # model inputs
        if self.episode.current_frame is None:
            model_input.state = self.state()
        else:
            model_input.state = self.episode.current_frame

        model_input.hidden = self.hidden

        model_input = self.process_detr_input(model_input)

        model_input.action_probs = self.last_action_probs

        if 'SP' in self.model_name:
            model_input.glove = self.glove[self.episode.target_object]

        if 'Memory' in self.model_name:
            if self.model_name.startswith('VR'):
                state_length = 64 * 7 * 7
            else:
                state_length = self.hidden_state_sz

            if len(self.episode.state_reps) == 0:
                model_input.states_rep = torch.zeros(1, state_length)
            else:
                model_input.states_rep = torch.stack(self.episode.state_reps)

            if 'State' in self.model_name:
                dim_obs = 3136
            else:
                dim_obs = 512
            if len(self.episode.obs_reps) == 0:
                model_input.obs_reps = torch.zeros(1, dim_obs)
            else:
                model_input.obs_reps = torch.stack(self.episode.obs_reps)

            if len(self.episode.state_memory) == 0:
                model_input.states_memory = torch.zeros(1, state_length)
            else:
                model_input.states_memory = torch.stack(self.episode.state_memory)

            if len(self.episode.action_memory) == 0:
                model_input.action_memory = torch.zeros(1, 6)
            else:
                model_input.action_memory = torch.stack(self.episode.action_memory)

            model_input.states_rep = torch.FloatTensor(model_input.states_repd).to(self.device)
            model_input.states_memory = torch.FloatTensor(model_input.states_memory).to(self.device)
            model_input.action_memory = torch.FloatTensor(model_input.action_memory).to(self.device)
            model_input.obs_reps = torch.FloatTensor(model_input.obs_reps).to(self.device)

        return model_input, self.model.forward(model_input, model_options)

    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        state = torch.Tensor(frame)
        return state.to(self.device)

    def reset_hidden(self):
        if 'SingleLayerLSTM' not in self.model_name:
            self.hidden = (
                torch.zeros(2, 1, self.hidden_state_sz).to(self.device),
                torch.zeros(2, 1, self.hidden_state_sz).to(self.device),
            )
        else:
            self.hidden = (
                torch.zeros(1, 1, self.hidden_state_sz).to(self.device),
                torch.zeros(1, 1, self.hidden_state_sz).to(self.device),
            )

        self.last_action_probs = torch.zeros((1, self.action_space)).to(self.device)

    def repackage_hidden(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        self.last_action_probs = self.last_action_probs.detach()

    def state(self):
        return self.preprocess_frame(self.episode.state_for_agent())

    def exit(self):
        pass

    def process_detr_input(self, model_input):
        # process detection features from DETR detector
        current_detection_feature = self.episode.current_detection_feature()

        zero_detect_feats = np.zeros_like(current_detection_feature)
        ind = 0
        for cate_id in range(len(self.targets) + 2):
            cate_index = current_detection_feature[:, 257] == cate_id
            if cate_index.sum() > 0:
                index = current_detection_feature[cate_index, 256].argmax(0)
                zero_detect_feats[ind, :] = current_detection_feature[cate_index, :][index]
                ind += 1
        current_detection_feature = zero_detect_feats

        current_detection_feature[current_detection_feature[:, 257] == (len(self.targets) + 1)] = 0

        detection_inputs = {
            'features': current_detection_feature[:, :256],
            'scores': current_detection_feature[:, 256],
            'labels': current_detection_feature[:, 257],
            'bboxes': current_detection_feature[:, 260:],
            'target': self.targets.index(self.episode.target_object),
        }

        # generate target indicator array based on detection results labels
        target_embedding_array = np.zeros((detection_inputs['features'].shape[0], 1))
        target_embedding_array[
            detection_inputs['labels'][:] == (self.targets.index(self.episode.target_object) + 1)] = 1
        detection_inputs['indicator'] = target_embedding_array

        detection_inputs = self.dict_toFloatTensor(detection_inputs)

        model_input.detection_inputs = detection_inputs

        return model_input


    def dict_toFloatTensor(self, dict_input):
        '''Convert all values in dict_input to float tensor

        '''
        for key in dict_input:
            dict_input[key] = torch.FloatTensor(dict_input[key]).to(self.device)

        return dict_input
