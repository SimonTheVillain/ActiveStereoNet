import cv2
import os
import numpy as np

from torch.utils.data import Dataset


#todo: this! + the same thing for a rendered dataset!!!!!!
class StructureCoreCapturedDataset(Dataset):

    def __init__(self, data_root, phase, crop_size=None, halfres=False):

        super(StructureCoreCapturedDataset, self).__init__()

        self.crop_size = crop_size
        self.data_root = data_root
        self.phase = phase
        self.halfres = halfres

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

        assert self.__len__() > 0, f"Captured dataset error: No sequences in {data_root}"

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

        if self.crop_size is not None:
            irl = irl[:self.crop_size[1], :self.crop_size[0]]
            irr = irr[:self.crop_size[1], :self.crop_size[0]]

        irl = np.expand_dims(irl, 0)
        irr = np.expand_dims(irr, 0)

        return irl, irr


#todo: this! + the same thing for a rendered dataset!!!!!!
class StructureCoreRenderedDataset(Dataset):

    def __init__(self, data_root, phase, crop_size=None, halfres=False, enable_gt_disp=True):

        super(StructureCoreRenderedDataset, self).__init__()

        self.enable_gt_disp = enable_gt_disp
        self.crop_size = crop_size
        self.data_root = data_root
        self.phase = phase
        self.halfres = halfres
        self.baseline = 0.07501
        self.focal = 1.1154399414062500e+03


        self.src_res = (1401, 1001)
        self.src_cxy = (700, 500)
        self.tgt_res = (1216, 896)
        self.tgt_cxy = (604, 457)
        self.readout_rect = (self.src_cxy[0]-self.tgt_cxy[0], self.src_cxy[1]-self.tgt_cxy[1],
                             self.tgt_res[0], self.tgt_res[1])
        #self.transform = transform

        self.keys = []

        files = os.listdir(data_root)
        if len(files) == 0:
            print(f"no files in {data_root}")
            exit()
        #print(files)
        keys = []
        for file in files:
            if os.path.isfile(f"{data_root}/{file}"):
                keys.append(file.split("_")[0])

        self.keys = list(set(keys))

        if len(self.keys) == 0:
            print(f"no valid files in {data_root}")
            exit()

        if phase == "train":
            self.idx_from = 0
            self.idx_to = int(len(self.keys) * 0.95)

        if phase == "val":
            self.idx_from = int(len(self.keys) * 0.95)
            self.idx_to = len(self.keys)

    def __len__(self):
        return self.idx_to - self.idx_from

    def __getitem__(self, idx):
        idx += self.idx_from
        key = self.keys[idx]

        rr = self.readout_rect

        filename = os.path.join(self.data_root, f"{key}_left.jpg")
        irl = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if irl is None:
            return self.__getitem__(np.random.randint(0, self.__len__()))
        irl = irl[rr[1]: rr[1] + rr[3], rr[0]: rr[0] + rr[2]]
        assert irl is not None, f"Does file {filename} exist?"
        filename = os.path.join(self.data_root, f"{key}_right.jpg")
        irr = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if irr is None:
            return self.__getitem__(np.random.randint(0, self.__len__()))

        irr = irr[rr[1]: rr[1] + rr[3], rr[0]: rr[0] + rr[2]]
        assert irr is not None, f"Does file {filename} exist?"

        filename = os.path.join(self.data_root, f"{key}_left_d.exr")
        d = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if d is None:
            return self.__getitem__(np.random.randint(0, self.__len__()))
        d = d[rr[1]: rr[1] + rr[3], rr[0]: rr[0] + rr[2]]
        assert d is not None, f"Does file {filename} exist?"

        disp = self.baseline * self.focal / d

        if self.halfres:
            irl = cv2.resize(irl, (int(irl.shape[1] / 2), int(irl.shape[0] / 2)))
            irr = cv2.resize(irr, (int(irr.shape[1] / 2), int(irr.shape[0] / 2)))

        irl = irl.astype(np.float32) * (1.0 / 255.0)
        irr = irr.astype(np.float32) * (1.0 / 255.0)

        channel_weights = np.random.random(3) * 2
        channel_weights = channel_weights / (np.sum(channel_weights) + 0.01)

        irl = irl[:, :, 0] * channel_weights[0] + \
               irl[:, :, 1] * channel_weights[1] + \
               irl[:, :, 2] * channel_weights[2]

        irr = irr[:, :, 0] * channel_weights[0] + \
               irr[:, :, 1] * channel_weights[1] + \
               irr[:, :, 2] * channel_weights[2]



        if self.crop_size is not None:
            irl = irl[:self.crop_size[1], :self.crop_size[0]]
            irr = irr[:self.crop_size[1], :self.crop_size[0]]
            disp = disp[:self.crop_size[1], :self.crop_size[0]]


        irl = np.expand_dims(irl, 0)
        irr = np.expand_dims(irr, 0)
        disp = np.expand_dims(disp, 0)

        if self.enable_gt_disp:
            return irl, irr, disp
        else:
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