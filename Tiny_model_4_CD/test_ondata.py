import torch
from dataset.dataset import MyDataset
import tqdm
from torch.utils.data import DataLoader
from metrics.metric_tool import ConfuseMatrixMeter
from models.change_classifier import ChangeClassifier
from models.layers import BBoxRegressor
from test_single_img import calculate_accuracy_score 
import argparse

def parse_arguments():
    # Argument Parser creation
    parser = argparse.ArgumentParser(
        description="Parameter for data analysis, data cleaning and model training."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        help="data path",
        default="/home/codegoni/aerial/WHU-CD-256/WHU-CD-256",
    )
    parser.add_argument(
        "--modelpath",
        type=str,
        help="model path",
    )

    parsed_arguments = parser.parse_args()
    
    return parsed_arguments

if __name__ == "__main__":

    # Parse arguments:
    args = parse_arguments()

    # Initialisation of the dataset
    data_path = args.datapath 
    dataset = MyDataset(data_path, "test")
    test_loader = DataLoader(dataset, batch_size=1)

    # Initialisation of the model and print model stat
    model = ChangeClassifier()
    modelpath = args.modelpath
    model.load_state_dict(torch.load(modelpath))

    # Print the number of model parameters 
    param_tot = sum(p.numel() for p in model.parameters())
    print()
    print("Number of model parameters {}".format(param_tot))
    print()

    # Set evaluation mode and cast the model to the desidered device
    model.eval()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)

    # loop to evaluate the model and print the metrics
    l1_loss = 0.0
    criterion = torch.nn.L1Loss()
    bb_regressor = BBoxRegressor()
    average_accuracy_score

    with torch.no_grad():
        for (reference, testimg), mask in tqdm.tqdm(test_loader):
            reference = reference.to(device).float()
            testimg = testimg.to(device).float()

            mask = mask.float()
            label_boxes = bb_regressor.get_bboxes(mask, 10)

            # pass refence and test in the model
            generated_boxes = model(reference, testimg)
            
            # compute the loss for the batch and backpropagate
            generated_boxes = generated_boxes.to("cpu")
            l1_loss += criterion(generated_boxes, label_boxes))

            # calculate accuracy score (bounding box overlap)
            accuracy_score = calculate_accuracy_score(generated_boxes, label_boxes)
            average_accuracy_score += accuracy_score

        average_accuracy_score = accuracy_score /= len(test_loader)
        l1_loss /= len(test_loader)
        print("Test summary")
        print("Loss is {}".format(l1_loss))
        print("Accuracy score is {}".format(average_accuracy_score))
        print()
