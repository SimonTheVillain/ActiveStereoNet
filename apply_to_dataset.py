import torch
import os
import cv2
import numpy as np
import re


model = torch.load("trained_models/train_structure_unity_pretrain_1_chk.pt")
model = torch.load("trained_models/train_structure_unity_full_supervision_2.pt")
rendered = True
half_res = True
focal = 1
baseline = 1
if rendered:

    src_res = (1401, 1001)
    src_cxy = (700, 500)
    tgt_res = (1216, 896)
    tgt_cxy = (604, 457)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03

    rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])
    path = "/media/simon/ssd_datasets/datasets/structure_core_unity_test"

    path = "/media/simon/LaCie/datasets/structure_core_unity_test"
    path_out = "/media/simon/ssd_datasets/datasets/structure_core_unity_test_results/ActiveStereoNet"
    inds = os.listdir(path)
    inds  = [re.search(r'\d+', s).group() for s in inds]
    inds = set(inds)
    inds = list(inds)
    inds.sort()
    paths = []
    for ind in inds:
        paths.append((ind, path + f"/{ind}_left.jpg", path + f"/{ind}_right.jpg"))
else:
    pass

with torch.no_grad():
    for ind, pl, pr in paths:
        p = pl
        irl = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        irl = cv2.cvtColor(irl, cv2.COLOR_RGB2GRAY)
        irl = irl[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]]
        irl = irl.astype(np.float32) * (1.0 / 255.0)

        p = pr

        irr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        irr = cv2.cvtColor(irr, cv2.COLOR_RGB2GRAY)
        irr = irr[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]]
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

        result = ref_pred.cpu()[0, 0, :, :].numpy()
        #result = coresup_pred.cpu()[0, 0, :, :].numpy()

        p = path_out + f"/{int(ind):05d}.exr"
        cv2.imshow("result", result * (1.0 / 50.0))
        cv2.imwrite(p, result)
        cv2.waitKey(1)
        print(p)


