import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Data.structure_core import StructureCoreCapturedDataset, StructureCoreRenderedDataset
from Models.ActiveStereoNet import ActiveStereoNet
from Losses.supervise import *
import cv2
import math
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--loss", dest="loss_type", action="store",
                        help="One of the three loss types: fully_supervised, classification and active_stereo",
                        default="active_stereo")
    parser.add_argument("-p", "--path", dest="dataset_path", action="store",
                        help="Path to dataset.",
                        default="")
    parser.add_argument("-chk", "--load_checkpoint", dest="load_checkpoint", action="store",
                        help="Name of the model to be loaded.",
                        default="")
    parser.add_argument("-e", "--experiment_name", dest="experiment_name", action="store",
                        help="Name of the current experiment.",
                        default="")
    parser.add_argument("-ss", "--step_scale", dest="step_scale", action="store",
                        help="How many times of the basic epochs/steps should the training procedure use?",
                        type=int,
                        default=4)

    parser.add_argument("-s", "--scale", dest="scale", action="store",
                        help="divider by which the image is downsampled.",
                        type=int,
                        default=1)

    parser.add_argument("-d", "--debug", dest="debug_output", action="store",
                        help="Enables debug output for our training.",
                        type=bool,
                        default=False)

    parser.add_argument("-bs", "--batch_size", dest="batch_size", action="store",
                        help="Batch size",
                        type=int,
                        default=2)

    parser.add_argument("--dataset_type", dest="dataset_type", action="store",
                        help="Type of the dataset \"rendered\" or \"captured\" for the structure core data.",
                        type=str,
                        default="captured")
    args = parser.parse_args()


    loss_type = "classification" #"fully_supervised" "classification" "active_stereo" "full_supervision_classification
    loss_type = args.loss_type
    #fully_supervised = True

    dataset_path = args.dataset_path
    if dataset_path == "":
        print("error! provide dataset_path (--path ...)")
        return
    experiment_name = args.experiment_name
    batch_size = args.batch_size
    num_workers = 8
    crop_size = [1216, 896]
    crop_size = [crop_size[0] // args.scale, crop_size[1] // args.scale]
    half_res = True
    if half_res:
        crop_size = [int(crop_size[0] / 2), int(crop_size[1] / 2)]
    max_disp = 144
    scale_factor = 8
    lr_init = 1e-4
    #lr_init = 1e-4 * 0.5**4
    scheduler_gamma = 0.5
    step_scale = args.step_scale
    scheduler_milestones = [int(20000 * step_scale),
                            int(30000 * step_scale),
                            int(40000 * step_scale),
                            int(50000 * step_scale)]
    overall_steps = int(60000 * step_scale)

    model = ActiveStereoNet(max_disp, scale_factor, crop_size, ch_in=1)
    if not args.load_checkpoint == "":
        model = torch.load(f"trained_models/{args.load_checkpoint}")
        #model.img_shape = [304 * 2, 224 * 2]  # TODO: remove this debug at early possibility
        #model.CoarseNet.img_shape = [304 * 2, 224 * 2]  # TODO: remove this debug at early possibility
    model = model.cuda()

    crit = XTLoss(max_disp, ch_in=1)
    #crit = RHLoss(max_disp)

    if loss_type == "active_stereo":
        if args.dataset_type == "captured":
            datasets = {x: StructureCoreCapturedDataset(dataset_path, phase=x, halfres=half_res, crop_size=crop_size)
                           for x in ['train', 'val']}
        if args.dataset_type == "unity":
            datasets = {x: StructureCoreRenderedDataset(dataset_path, phase=x, halfres=half_res, crop_size=crop_size,
                                                        sequences=True)
                           for x in ['train', 'val']}

    if loss_type in ["fully_supervised", "classification"]:
        datasets = {x: StructureCoreRenderedDataset(dataset_path, phase=x, halfres=half_res, crop_size=crop_size,
                                                    sequences=True)
                    for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
                   for x in ['train', 'val']}


    #optimizer = optim.RMSprop(model.parameters(), lr=lr_init, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestones,
                                               gamma=scheduler_gamma)

    writer = SummaryWriter(f"tensorboard/{experiment_name}")

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    epochs = overall_steps // dataset_sizes["train"] + 1
    step = 0
    loss_min = 10000000
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs - 1}')
        print('-' * 10)

        for phase in ["train", "val"]:
            print(phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            loss_accu = 0
            loss_accu_sub = 0
            for i_batch, sampled_batch in enumerate(dataloaders[phase]):
                if len(sampled_batch) == 2:
                    irl, irr = sampled_batch
                    irl, irr = irl.cuda(), irr.cuda()
                else:
                    irl, irr, disp_gt = sampled_batch
                    irl, irr, disp_gt = irl.cuda(), irr.cuda(), disp_gt.cuda()


                if phase == "train":
                    optimizer.zero_grad()
                    disp_pred_left, coarsedisp_pred_left, presoftmax = model(irl, irr)
                    #debug_presoftmax.retain_grad() # debug remove
                    if loss_type == "fully_supervised":
                        loss = torch.abs(disp_gt - disp_pred_left).mean() * 1.0
                        #loss += torch.abs(disp_gt - coarsedisp_pred_left).mean()
                    if loss_type == "classification":
                        disp_gt_2 = F.interpolate(disp_gt, (disp_gt.shape[2] // 8, disp_gt.shape[3] // 8 ))
                        disp_gt_2 = ((disp_gt_2 + 4) // 8)
                        disp_gt_2 = disp_gt_2.squeeze(1).type(torch.int64).clamp(0, max_disp//8 - 1)
                        presoftmax = presoftmax.squeeze(1)
                        loss = F.cross_entropy(presoftmax, disp_gt_2, reduction="mean")
                        loss += torch.abs(disp_gt - disp_pred_left).mean() * 1.0
                    if loss_type == "active_stereo":
                        loss = crit(irl, irr, disp_pred_left)

                    #disp_pred_left.retain_grad()# todo: debug remove
                    loss.backward()
                    #print(debug_presoftmax.grad.abs().max())
                    #print(disp_pred_left.grad.abs().max())
                    #print(debug_presoftmax.grad[0, 0, :, 100, 100])
                    optimizer.step()
                    scheduler.step()
                    loss = loss.item()

                else:
                    with torch.no_grad():
                        disp_pred_left, coarsedisp_pred_left, presoftmax = model(irl, irr)

                        if loss_type == "fully_supervised":
                            loss = torch.abs(disp_gt - disp_pred_left).mean() * 1.0
                            #loss += torch.abs(disp_gt - coarsedisp_pred_left).mean()
                        if loss_type == "classification":
                            disp_gt_2 = F.interpolate(disp_gt, (disp_gt.shape[2] // 8, disp_gt.shape[3] // 8 ))
                            disp_gt_2 = ((disp_gt_2 + 4) // 8) # + 4 for nearest interpolation
                            disp_gt_2 = disp_gt_2.squeeze(1).type(torch.int64).clamp(0, max_disp//8 - 1)
                            presoftmax = presoftmax.squeeze(1)
                            loss = F.cross_entropy(presoftmax, disp_gt_2, reduction="mean")

                            loss += torch.abs(disp_gt - disp_pred_left).mean() * 1.0
                        if loss_type == "active_stereo":
                            loss = crit(irl, irr, disp_pred_left)

                        loss = loss.item()

                assert not math.isnan(loss) and not math.isinf(loss), "NAN/INF found. ABORT!"
                loss_accu += loss
                loss_accu_sub += loss
                step += 1
                if i_batch % 100 == 99:


                    # todo: show reprojected
                    if args.debug_output:
                        cv2.imshow("irl", irl[0,0,:,:].detach().cpu().numpy())
                        cv2.imshow("irr", irr[0,0,:,:].detach().cpu().numpy())

                        if "disp_gt" in vars():
                            theta = torch.Tensor(
                                [[1, 0, 0],  # 控制左右，-右，+左
                                 [0, 1, 0]]  # 控制上下，-下，+上
                            )
                            theta = theta.repeat(irl.size()[0], 1, 1)
                            grid = F.affine_grid(theta, irl.size(), align_corners=True)  # enable old behaviour
                            # print(grid)
                            grid = grid.cuda()
                            dispmap_norm = disp_gt * 2 / disp_gt.shape[3] # times 2 because grid_sample normalizes between -1 and 1!

                            dispmap_norm = dispmap_norm.squeeze(1).unsqueeze(3)
                            # print(dispmap_norm.shape)
                            dispmap_norm = torch.cat((dispmap_norm, torch.zeros(dispmap_norm.size()).cuda()), dim=3)
                            # print(dispmap_norm.shape)
                            grid -= dispmap_norm

                            recon_img = F.grid_sample(irr, grid, align_corners=True)  # enable old behaviour

                            cv2.imshow("recon_img", recon_img[0,0,:,:].detach().cpu().numpy())

                            print(f"disp_gt.max {disp_gt.max()}")
                            print(f"disp_gt.min {disp_gt.min()}")
                            print(f"disp_gt.mean {disp_gt.mean()}")
                            disp_gt = disp_gt[0, 0, :, :]
                            disp_gt -= disp_gt.min()
                            disp_gt /= disp_gt.max() + 0.1
                            cv2.imshow("disp_gt", disp_gt.detach().cpu().numpy())

                        print(f"pred.max {disp_pred_left.max()}")
                        print(f"pred.min {disp_pred_left.min()}")
                        print(f"pred.mean {disp_pred_left.mean()}")
                        disp_pred_left = disp_pred_left[0, 0, :, :]
                        disp_pred_left -= disp_pred_left.min()
                        disp_pred_left /= disp_pred_left.max() + 0.1
                        cv2.imshow("disp_pred", disp_pred_left.detach().cpu().numpy())



                        print(f"coarsepred.max {coarsedisp_pred_left.max()}")
                        print(f"coarsepred.min {coarsedisp_pred_left.min()}")
                        print(f"coarsepred.mean {coarsedisp_pred_left.mean()}")
                        coarsedisp_pred_left = coarsedisp_pred_left[0, 0, :, :]
                        coarsedisp_pred_left -= coarsedisp_pred_left.min()
                        coarsedisp_pred_left /= coarsedisp_pred_left.max() + 0.1
                        cv2.imshow("coarsedisp_pred_left", coarsedisp_pred_left.detach().cpu().numpy())

                        cv2.waitKey(100)


                    loss = loss_accu_sub / 100
                    loss_accu_sub = 0
                    print(f"step {step} with loss {loss}")
                    writer.add_scalar(f"{phase}/loss", loss, step)
            loss = loss_accu * batch_size / dataset_sizes[phase]
            print(f"Loss for full {phase} phase of epoch {epoch}: {loss}")
            writer.add_scalar(f"{phase}/loss_epoch", loss, step)

            if phase == "val":
                #storing checkpoint

                torch.save(model, f"trained_models/{experiment_name}_chk.pt")
                if loss < loss_min:
                    print("storing new best network")
                    loss_min = loss
                    torch.save(model, f"trained_models/{experiment_name}.pt")






if __name__ == "__main__":
    main()