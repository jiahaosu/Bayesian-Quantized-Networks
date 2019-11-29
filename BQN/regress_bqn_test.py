# system utilities
from __future__ import print_function
import os, datetime, argparse

# pytorch utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.datasets_boston_housing import Dataset_boston_housing

# computing utilities
import numpy as np
import math

# custom utilities
from BayesNets import BayesReNet
from tensorboardX import SummaryWriter

def main(args):
    ## Devices (CPU, single GPU or multiple GPU)

    # whether to use GPU (or CPU) 
    use_cuda  = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device: ", device)

    # whether to use multi-GPU (or single-GPU)
    multi_gpu = use_cuda and args.multi_gpu and torch.cuda.device_count() > 1
    num_gpus = (torch.cuda.device_count() if multi_gpu else 1) if use_cuda else 0
    print("# of GPUs: ", num_gpus)


    ## Paths (Dataset, Checkpoints, Statistics)
    
    # path to the folder of all datasets
    data_path = args.data_path
    if not os.path.exists(data_path): 
        os.makedirs(data_path)

    # path to the folder of specified dataset
    dataset = args.dataset
    assert dataset in ["Boston"], \
        "The specified dataset is not supported."
    print("Dataset: ", dataset)

    if args.dataset_path == "default":
        dataset_path = dataset
    else: # if args.dataset_path != "default":
        dataset_path = args.dataset_path

    data_path = os.path.join(data_path, dataset_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # path to the folder of all outputs
    outputs_path = args.outputs_path
    assert os.path.exists(outputs_path), \
        "The outputs folder does not exist."

    # path to the folder of specifed dataset
    outputs_path = os.path.join(outputs_path, dataset_path)
    assert os.path.exists(outputs_path), \
        "The outputs folder for the dataset does not exist."

    # create the name of the current network architecture
    outputs_path = os.path.join(outputs_path, args.network_name)
    assert os.path.exists(outputs_path), \
        "The outputs folder for the architecture does not exist."
    
    # path to the folder of current model
    outputs_path = os.path.join(outputs_path, 
        args.model_name + '_' + args.model_stamp)
    assert os.path.exists(outputs_path), \
        "The outputs folder for the specified model does not exist."

    # path to the folder of checkpoints
    model_path = os.path.join(outputs_path, args.model_path)
    assert os.path.exists(model_path), \
        "The models folder does not exist."

    # path to the folder of the evaluation statistics
    stats_path = os.path.join(outputs_path, args.stats_path)
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)


    ## Data formats and Dataloaders 
    # data format: batch_size(0) x intut_length (1)

    # batch size and the log intervals (0)
    batch_size  = args.batch_size

    if dataset == "Boston":
        test_dataset  = Dataset_boston_housing(
            '../data/Boston/boston_housing_nor_val.pkl')
    else: 
        raise NotImplementedError

    # number of worker for dataloaders 
    num_workers = min(num_gpus, 1) * 4

    test_loader  = torch.utils.data.DataLoader(dataset = test_dataset,
        batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = False)

    test_samples =  len(test_loader.dataset)
    print("# of validation samples: ", test_samples)
    

    ## Models (Multi-layer Perceptron or Convolutional Neural Networks)

    # quantization levels
    quantization  = args.quantization
    assert quantization in ["binary", "ternary", "quanternary"], \
        "The type of weights quantization is not supported."
    nat_per_param = {"binary": math.log(2), "ternary": math.log(3),
        "quanternary": math.log(4)}[quantization]

    # model architecture
    model_type = args.model_type
    assert model_type in ["ReNet"], \
        "The type of model archiecture is not supported."

    print("Model: ", model_type, "(Quantization: %s)" % quantization)

    # multi-layer perceptron
    if model_type == "ReNet":
        model = BayesReNet(input_length = 13, 
            use_bias = args.use_bias, quantization = quantization)
    else:
        raise NotImplementedError

    # number of parameters in the neural network
    (num_params, ) = model.num_params()
    print("# of parameters: ", num_params)

    # move the model to the device (CPU, single-GPU, multi-GPU) 
    model.to(device)
    if multi_gpu: model = nn.DataParallel(model)


    ## Main script for testing
    if args.eval_auto:
        if args.eval_best:
            model_file = os.path.join(model_path, 'training_best.pt')
        else: # if args.eval_last:
            model_file = os.path.join(model_path, 'training_last.pt')
    else: # if args.eval_spec:
        model_file = os.path.join(model_path, 'training_%d.pt' % args.eval_epoch)

    assert os.path.exists(model_file), \
        "The specified model is not found in the folder."

    checkpoint = torch.load(model_file)
    eval_epoch = checkpoint.get("epoch", args.eval_epoch)
    model.load_state_dict(checkpoint["model_state_dict"])

    # path to the file of the evaluation statistics
    stats_file = "stats_%d" % eval_epoch if (args.stats_file
        == "default") else args.stats_file 
    stats_file = os.path.join(stats_path, stats_file)

    # number of Monte-Carlo samples for evaluation
    mc_samples = args.mc_samples

    test_nll, test_rmse = 0., 0. # Probabilistic Propagation
    test_nll_map, test_rmse_map = 0., 0.
    test_nll_mcs, test_rmse_mcs = 0., 0.

    with torch.no_grad():
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # (1) Probabilistic Propagation
            (outputs, loss_nll) = model(data, target, mode = "analytic")
            loss_nll = torch.mean(loss_nll) # for multi-gpu

            rmse = torch.sum((outputs[0] - target) ** 2, dim = 1) ** 0.5

            test_nll  += loss_nll.item()
            test_rmse += torch.sum(rmse).item()

            # (2) Maximum A Posterior and point propagation
            (outputs, loss_nll) = model(data, target, mode = "MAP")
            loss_nll = torch.mean(loss_nll) # for multi-gpu

            rmse = torch.sum((outputs - target) ** 2, dim = 1) ** 0.5

            test_nll_map  += loss_nll.item()
            test_rmse_map += torch.sum(rmse).item()

            # (3) Monte Carlo Sampling and point propagation
            for s in range(mc_samples):
                (outputs_, loss_nll) = model(data, target, mode = "sampling")
                loss_nll = torch.mean(loss_nll) # for multi-gpu

                test_nll_mcs  += loss_nll.item() / mc_samples

                outputs = outputs_ / mc_samples if s == 0 \
                    else outputs + outputs_ / mc_samples

            rmse = torch.sum((outputs - target) ** 2, dim = 1) ** 0.5
            test_rmse_mcs += torch.sum(rmse).item()

        # (1) Probabilistic propagation (Analytical inference)
        test_nll  /= test_samples
        test_rmse /= test_samples 
        print("Epoch {} (Analytical) , Avg. NLL: {:.4f}, Avg. RMSE: {:4f}".format(
            eval_epoch, test_nll, test_rmse))

        # (2) Maximum A Posterior and point propagation
        test_nll_map  /= test_samples
        test_rmse_map /= test_samples 
        print("Epoch {} (Analytical) , Avg. NLL: {:.4f}, Avg. RMSE: {:4f}".format(
            eval_epoch, test_nll_map, test_rmse_map))

        # (3) Monte Carlo Sampling and point propagation
        test_nll_mcs  /= test_samples
        test_rmse_mcs /= test_samples 
        print("Epoch {} (Analytical) , Avg. NLL: {:.4f}, Avg. RMSE: {:4f}".format(
            eval_epoch, test_nll_mcs, test_rmse_mcs))

    # save the statistics 
    np.savez(stats_file, 
        test_rmse     = test_rmse,     test_nll     = test_nll,
        test_rmse_map = test_rmse_map, test_nll_map = test_nll_map,
        test_rmse_mcs = test_rmse_mcs, test_nll_mcs = test_nll_mcs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 
        "Training Bayesian Quantized Networks (BQN).")

    ## 1) Data format (Pytorch format)
    # batch size (0) x input_length (1)

    # batch size and log interval (0)
    parser.add_argument("--batch-size",  default =  4, type = int,
        help = "The batch size for training.")

    # image format: input_length(1)
    parser.add_argument("--input-length", default = 13, type = int,
        help = "The input length of each sample.")

    ## 2) Paths (Data, Checkpoints, Results and TensorboardX)

    # inputs:  data
    parser.add_argument("--dataset", default = "Boston", type = str,
        help = "The dataset used for training (options: Boston).")
    parser.add_argument("--data-path", default = "../data", type = str,
        help = "The path to the folder stroing the data.")
    parser.add_argument("--dataset-path", default = "default", type = str, 
        help = "The folder for the test dataset.")

    # outputs: checkpoints, statistics and tensorboard
    parser.add_argument("--outputs-path", default = "../outputs_BQN", type = str,
        help = "The path to the folder storing outputs from training.")
    parser.add_argument("--network-name", default = "default", type = str, 
        help = "The architecture model (to create the folder).")

    parser.add_argument("--model-path", default = "models", type = str,
        help = "The folder for all checkpoints in training.")
    parser.add_argument("--stats-path", default = "stats",  type = str,
        help = "The folder for the evaluation statistics.")
    parser.add_argument("--stats-file", default = "default", type = str, 
        help = "The file name for the evaluation statistics.")

    ## 3) Device (CPU, single GPU or multiple GPUs)
    
    # whether to use GPU for training
    parser.add_argument('--use-cuda', dest = 'use_cuda', action = 'store_true',
        help = 'Use GPU for training.')
    parser.add_argument('--no-cuda',  dest = 'use_cuda', action = 'store_false', 
        help = "Do not use GPU for training.")
    parser.set_defaults(use_cuda = True)

    # whether to use multi-GPU for training 
    parser.add_argument('--multi-gpu',  dest = 'multi_gpu', action = 'store_true',
        help = 'Use multiple GPUs for training.')
    parser.add_argument('--single-gpu', dest = 'multi_gpu', action = 'store_false',
        help = 'Do not use multiple GPU for training.')
    parser.set_defaults(multi_gpu = False)

    ## 4) Models (RegressNet)
    parser.add_argument("--quantization", default = "binary", type = str, 
        help = "The type of quantization (options: binary/ternary/quanternary).")
    parser.add_argument("--model-type", default = "ReNet", type = str,
        help = "The type of the model (options: ReNet).")

    parser.add_argument("--use-bias", dest = "use_bias", action = "store_true", 
        help = "Use bias in all layers of the model.")
    parser.add_argument("--no-bias",  dest = "use_bias", action = "store_false", 
        help = "Do not use bias in all layers of the model.")
    parser.set_defaults(use_bias = True)

    parser.add_argument("--model-name",  default = "default", type = str,
        help = "The model name (to create the associated folder).")
    parser.add_argument("--model-stamp", default = "default", type = str,
        help = "The time stamp of the model (as a suffix to the its name).")

    ## Hyperparameters for evaluation
    parser.add_argument('--eval-auto', dest = 'eval_auto', action = 'store_true', 
        help = 'Evaluate the best or the last model.')
    parser.add_argument('--eval-spec', dest = 'eval_auto', action = 'store_false', 
        help = 'Evaluate the model of specified epoch')
    parser.set_defaults(eval_auto = True)

    # if eval_auto is True (--eval-auto)
    parser.add_argument('--eval-best', dest = 'eval_best', action = 'store_true',
        help = 'Evaluate the best model (in term of validation loss).')
    parser.add_argument('--eval-last', dest = 'eval_best', action = 'store_false',
        help = 'Evaluate the last model (in term of training epoch).')
    parser.set_defaults(eval_best = False)

    # if eval_auto is False (--eval-spec)
    parser.add_argument('--eval-epoch', default = 400, type = int, 
        help = 'Evaluate the model of specified epoch.')

    parser.add_argument("--mc-samples", default = 10, type = int, 
        help = "The number of Monte-Carlo samples to evaluate the model.")

    main(parser.parse_args())