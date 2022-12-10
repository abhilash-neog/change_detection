import argparse
import os
import shutil


import dataset.dataset as dtset
import torch
import numpy as np
import random
from models.change_classifier import ChangeClassifier as Model
from models.layers import BBoxRegressor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def parse_arguments():
    # Argument Parser creation
    parser = argparse.ArgumentParser(
        description="Parameter for data analysis, data cleaning and model training."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        help="data path",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        help="log path",
    )

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')

    parsed_arguments = parser.parse_args()

    # create log dir if it doesn't exists
    if not os.path.exists(parsed_arguments.log_path):
        os.mkdir(parsed_arguments.log_path)

    dir_run = sorted(
        [
            filename
            for filename in os.listdir(parsed_arguments.log_path)
            if filename.startswith("run_")
        ]
    )

    if len(dir_run) > 0:
        num_run = int(dir_run[-1].split("_")[-1]) + 1
    else:
        num_run = 0
    parsed_arguments.log_path = os.path.join(
        parsed_arguments.log_path, "run_%04d" % num_run + "/"
    )

    return parsed_arguments


def train(
    dataset_train,
    dataset_val,
    model,
    optimizer,
    scheduler,
    logpath,
    writer,
    epochs,
    save_after,
    device
):

    model = model.to(device)

    def evaluate(reference, testimg, boxes):
        # Evaluating the model:
        generated_boxes = model(reference.cuda(), testimg.cuda())

        # Loss gradient descend step:
        loss = torch.nn.L1Loss()

        it_loss = loss(generated_boxes, boxes)
        return it_loss

    def training_phase(epc):
        print("Epoch {}".format(epc))
        model.train()
        epoch_loss = 0.0
        bbox_regressor = BBoxRegressor()
        for (reference, testimg), mask in dataset_train:

            #convert label mask to bounding boxes
            label_boxes = bbox_regressor.get_bboxes(mask, 10)

            # Reset the gradients:
            optimizer.zero_grad()

            # Loss gradient descend step:
            it_loss = evaluate(reference, testimg, label_boxes.cuda())
            it_loss.backward()
            optimizer.step()

            # Track metrics:
            epoch_loss += it_loss.to("cpu").detach().numpy()
            ### end of iteration for epoch ###

        epoch_loss /= len(dataset_train)

        #########
        print("Training phase summary")
        print("Loss for epoch {} is {}".format(epc, epoch_loss))

        ### Save the model ###
        if epc % save_after == 0:
            torch.save(
                model.state_dict(), os.path.join(logpath, "model_{}.pth".format(epc))
            )

    def validation_phase(epc):
        model.eval()
        epoch_loss_eval = 0.0
        bbox_regressor = BBoxRegressor()
        with torch.no_grad():
            for (reference, testimg), mask in dataset_val:
                label_boxes = bbox_regressor.get_bboxes(mask, 10)

                epoch_loss_eval += evaluate(reference,
                                            testimg, label_boxes.cuda())

        epoch_loss_eval /= len(dataset_val)
        print("Validation phase summary")
        print("Loss for epoch {} is {}".format(epc, epoch_loss_eval))

    for epc in range(epochs):
        training_phase(epc)
        validation_phase(epc)
        # scheduler step
        scheduler.step()


def run():

    # set the random seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Parse arguments:
    args = parse_arguments()

    # Initialize tensorboard:
    writer = SummaryWriter(log_dir=args.log_path)

    # Inizialitazion of dataset and dataloader:
    trainingdata = dtset.MyDataset(args.datapath, "train")
    validationdata = dtset.MyDataset(args.datapath, "val")
    data_loader_training = DataLoader(trainingdata, batch_size=8, shuffle=True)
    data_loader_val = DataLoader(validationdata, batch_size=8, shuffle=True)

    # device setting for training
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    print(f'Current Device: {device}\n')

    # Initialize the model
    model = Model()
    restart_from_checkpoint = False
    model_path = None
    if restart_from_checkpoint:
        model.load_state_dict(torch.load(model_path))
        print("Checkpoint succesfully loaded")

    # print number of parameters
    parameters_tot = 0
    for nom, param in model.named_parameters():
        # print (nom, param.data.shape)
        parameters_tot += torch.prod(torch.tensor(param.data.shape))
    print("Number of model parameters {}\n".format(parameters_tot))

    # Optimizer with tuned parameters for LEVIR-CD
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00356799066427741,
                                  weight_decay=0.009449677083344786, amsgrad=False)

    # scheduler for the lr of the optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100)

    # copy the configurations
    _ = shutil.copytree(
        "./models",
        os.path.join(args.log_path, "models"),
    )

    train(
        data_loader_training,
        data_loader_val,
        model,
        optimizer,
        scheduler,
        args.log_path,
        writer,
        epochs=20,#epochs=100,
        save_after=1,
        device=device
    )
    writer.close()


if __name__ == "__main__":
    run()
