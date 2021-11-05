import torch
import os
import cv2
import numpy as np
import re
from pathlib import Path

from Losses.supervise import *


#model = torch.load("trained_models/train_structure_unity_pretrain_1_chk.pt")
#model = torch.load("trained_models/train_structure_unity_full_supervision_2.pt")

rendered = False
half_res = True
focal = 1
baseline_left_right = 0.07501
baseline_to_projector = 0.0634
if rendered:

    src_res = (1401, 1001)
    src_cxy = (700, 500)
    tgt_res = (1216, 896)
    tgt_cxy = (604, 457)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03

    rrr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])
    rrl = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])
    path = "/media/simon/ssd_datasets/datasets/structure_core_unity_test"

    path = "/media/simon/LaCie/datasets/structure_core_unity_test"
    path_out = "/media/simon/ssd_datasets/datasets/structure_core_unity_test_results/ActiveStereoNet"

    path = "/home/simon/datasets/structure_core_unity_test"
    path_out = "/home/simon/datasets/structure_core_unity_test_results/ActiveStereoNet"
    model = torch.load("trained_models/train_unity_new.pt")

    path = "/home/simon/datasets/structure_core_unity_test"
    path_out = "/home/simon/datasets/structure_core_unity_test_results/ActiveStereoNetFull"
    model = torch.load("trained_models/pretrain_sequences_new2.pt")
    inds = os.listdir(path)
    inds  = [re.search(r'\d+', s).group() for s in inds]
    inds = set(inds)
    inds = list(inds)
    inds.sort()
    paths = []
    for ind in inds:
        pout = path_out + f"/{int(ind):05d}.exr"
        paths.append((path + f"/{ind}_left.jpg", path + f"/{ind}_right.jpg", pout))
else:
    tgt_res = (1216, 896)
    tgt_cxy = (604, 457)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03

    rrr = (0, 0, tgt_res[0], tgt_res[1])
    rrl = (tgt_res[0], 0, tgt_res[0], tgt_res[1])
    path = "/media/simon/ssd_datasets/datasets/structure_core_photoneo_test"
    path_out = "/media/simon/ssd_datasets/datasets/structure_core_photoneo_test_results/ActiveStereoNet"

    path = "/home/simon/datasets/structure_core_photoneo_test"
    path_out = "/home/simon/datasets/structure_core_photoneo_test_results/ActiveStereoNet"
    model = torch.load("trained_models/train_real_world_new6.pt")
    folders = os.listdir(path)
    scenes = [x for x in folders if os.path.isdir(Path(path) / x)]

    paths = []
    for scene in scenes:
        tgt_path = Path(path_out) / scene
        if not os.path.exists(tgt_path):
            os.mkdir(tgt_path)
        for i in range(4):
            pr = Path(path) / scene / f"ir{i}.png"
            pl = Path(path) / scene / f"ir{i}.png"
            tgt_pth = Path(tgt_path) / f"{i}.exr"
            paths.append((str(pl), str(pr), str(tgt_pth)))

show_reprojection = False
if show_reprojection:
    max_disp = 144
    crit = XTLoss(max_disp, ch_in=1)
with torch.no_grad():
    for pl, pr, pout in paths:
        p = pl
        irl = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if len(irl.shape) == 3:
            # the rendered images are 3 channel bgr
            irl = cv2.cvtColor(irl, cv2.COLOR_BGR2GRAY)
        else:
            # the rendered images are 16
            irl = irl / 255.0
        irl = irl[rrl[1]:rrl[1] + rrl[3], rrl[0]:rrl[0] + rrl[2]]

        irl = irl.astype(np.float32) * (1.0 / 255.0)

        p = pr

        irr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if len(irr.shape) == 3:
            # the rendered images are 3 channel bgr
            irr = cv2.cvtColor(irr, cv2.COLOR_BGR2GRAY)
        else:
            # the rendered images are 16
            irr = irr / 255.0
        irr = irr[rrr[1]:rrr[1] + rrr[3], rrr[0]:rrr[0] + rrr[2]]
        irr = irr.astype(np.float32) * (1.0 / 255.0)

        if half_res:
            irl = cv2.resize(irl, (int(irl.shape[1] / 2), int(irl.shape[0] / 2)))
            irr = cv2.resize(irr, (int(irr.shape[1] / 2), int(irr.shape[0] / 2)))

        irl = irl[:448, :608]
        irr = irr[:448, :608]
        cv2.imshow("irleft", irl)
        cv2.imshow("irright", irr)
        irl = torch.tensor(irl).cuda().unsqueeze(0).unsqueeze(0)
        irr = torch.tensor(irr).cuda().unsqueeze(0).unsqueeze(0)

        ref_pred, coresup_pred, presoftmax = model(irl, irr)

        if show_reprojection:
            crit(irl, irr, ref_pred, True)

        result = ref_pred.cpu()[0, 0, :, :].numpy()
        #result = coresup_pred.cpu()[0, 0, :, :].numpy()

        p = pout#path_out + f"/{int(ind):05d}.exr"
        cv2.imshow("result", result * (1.0 / 50.0))
        result = result * baseline_to_projector / baseline_left_right
        cv2.imwrite(p, result)
        cv2.waitKey(1)
        print(p)


