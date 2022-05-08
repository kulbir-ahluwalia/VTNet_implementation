import os
import json
import numpy as np
from torch.utils.data import Dataset
from .constants import AI2THOR_TARGET_CLASSES


class PreVTDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        assert os.path.exists(data_dir), f"{data_dir} does not exist."

        self.data_dir = data_dir
        self.targets_index = [i for i, item in enumerate(AI2THOR_TARGET_CLASSES[60]) if item in AI2THOR_TARGET_CLASSES[22]]
        self.annotation_file = os.path.join(self.data_dir, 'annotation_{}.json'.format(split))
        with open(self.annotation_file, "r") as rf:
            self.annotations = json.load(rf)


    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        location = annotation["location"]
        target = annotation["target"]
        optimal_action = annotation["optimal_action"]
        annotation_path = os.path.join(self.data_dir, 'data', '{}.npz'.format(location))
        data = np.load(annotation_path)

        global_feature = data['resnet18_feature']

        features = data['detr_feature'][:, :256]
        scores = data['detr_feature'][:, 256]
        labels = data['detr_feature'][:, 257]
        bboxes = data['detr_feature'][:, 260:]

        # generate target indicator array based on detection results labels
        target_embedding_array = np.zeros((data['detr_feature'].shape[0], 1))
        target_embedding_array[labels[:] == (AI2THOR_TARGET_CLASSES[22].index(target) + 1)] = 1

        local_feature = {
            "features": features,
            "scores": scores,
            "labels": labels,
            "bboxes": bboxes,
            "indicator": target_embedding_array,
            "locations": location,
            "targets": target,
            "idx": idx,
        }

        return global_feature, local_feature, optimal_action
