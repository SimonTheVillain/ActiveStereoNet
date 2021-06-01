import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Data.structure_core import StructureCoreCapturedDataset
from Models.ActiveStereoNet import ActiveStereoNet
from Losses.supervise import *


def main():
    dataset_path = "/home/simon/datasets/structure_core/sequences_combined"
    dataset_path = "/media/simon/ext_ssd/datasets/structure_core/sequences_combined"

    experiment_name = "test3"
    batch_size = 4#2 in the example
    num_workers = 8
    crop_size = [608, 448]#[1216, 896]# [960, 540] original sceneflow resolution [1280, 720] would be for activestereonet
    crop_size = [1216, 896]
    half_res = True
    if half_res:
        crop_size = [int(crop_size[0] / 2), int(crop_size[1] / 2)]
    max_disp = 144
    scale_factor = 8

    lr_init = 1e-3
    scheduler_gamma = 0.5
    step_scale = 2
    scheduler_milestones = [int(20000 * step_scale),
                            int(30000 * step_scale),
                            int(40000 * step_scale),
                            int(50000 * step_scale)]
    overall_steps = int(60000 * step_scale)

    model = ActiveStereoNet(max_disp, scale_factor, crop_size, ch_in=1)
    model = model.cuda()

    crit = XTLoss(max_disp, ch_in=1)
    #crit = RHLoss(max_disp)

    datasets = {x: StructureCoreCapturedDataset(dataset_path, phase=x, halfres=half_res)
                   for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
                   for x in ['train', 'val']}


    #optimizer = optim.RMSprop(model.parameters(), lr=lr_init)
    optimizer = optim.Adam(model.parameters(), lr=lr_init)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestones,
                                               gamma=scheduler_gamma)

    writer = SummaryWriter(f"tensorboard/{experiment_name}")

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    epochs = int(overall_steps / dataset_sizes["train"])
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

                if phase == "train":
                    optimizer.zero_grad()

                    disp_pred_left = model(irl, irr)

                    loss = crit(irl, irr, disp_pred_left)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    loss = loss.item()

                else:
                    with torch.no_grad():
                        disp_pred_left = model(irl, irr)

                        loss = crit(irl, irr, disp_pred_left)
                        loss = loss.item()

                loss_accu += loss
                loss_accu_sub += loss
                step += 1
                if i_batch % 100 == 99:
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