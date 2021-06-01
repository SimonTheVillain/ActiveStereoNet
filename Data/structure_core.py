import cv2
import os
import numpy as np

from torch.utils.data import Dataset


#todo: this! + the same thing for a rendered dataset!!!!!!
class StructureCoreCapturedDataset(Dataset):

    def __init__(self, data_root, phase, halfres=False):

        super(StructureCoreCapturedDataset, self).__init__()

        self.data_root = data_root
        self.phase = phase
        self.halfres = halfres
        #self.transform = transform

        self.scene_paths = []

        potential_paths = os.listdir(data_root)
        potential_paths.sort()
        for path in potential_paths:
            path = os.path.join(data_root, path)
            if os.path.isdir(path):
                self.scene_paths.append(path)

        if phase == "train":
            self.idx_from = 0
            self.idx_to = int(len(self.scene_paths) * 0.95)

        if phase == "val":
            self.idx_from = int(len(self.scene_paths) * 0.95)
            self.idx_to = len(self.scene_paths)


    def __len__(self):
        return (self.idx_to - self.idx_from) * 4
    def __getitem__(self, idx):
        scene_idx = int(idx / 4) + self.idx_from
        frame_idx = idx % 4

        scene_path = self.scene_paths[scene_idx]

        filename = os.path.join(scene_path, f"ir{frame_idx}.png")
        ir = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        assert ir is not None, f"Does file {filename} exist?"
        if self.halfres:
            ir = cv2.resize(ir, (int(ir.shape[1] / 2), int(ir.shape[0] / 2)))

        ir = ir.astype(np.float32) * (1.0 / 65536.0)
        irl = ir[:, int(ir.shape[1] / 2):]
        irr = ir[:, :int(ir.shape[1] / 2)]

        irl = np.expand_dims(irl, 0)
        irr = np.expand_dims(irr, 0)

        return irl, irr



def test_dataset():
    dataset_path = "/media/simon/ext_ssd/datasets/structure_core/sequences_combined"
    split = "train" # as opposed to val
    dataset = StructureCoreCapturedDataset(dataset_path, split)
    for data in dataset:
        irl, irr = data
        cv2.imshow("irl", irl[0, :, :])
        cv2.imshow("irr", irr[0, :, :])
        cv2.waitKey()

if __name__ == "__main__":
    test_dataset()