# system utilities
from __future__ import print_function
import os, datetime, argparse

# pytorch utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# computing utilities
import numpy as np
import math

# custom utilities
from BayesNets import BayesNet, BayesConvNet, BayesVGGLike, BayesConvNet_, BayesVGGLike_

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

    # number of worker for dataloaders 
    num_workers = min(num_gpus, 1) * 4


    ## Paths (Dataset, Checkpoints, Statistics)
    
    # path to the folder of all datasets
    data_path = args.data_path
    if not os.path.exists(data_path): 
        os.makedirs(data_path)

    # path to the folder of specified dataset
    dataset = args.dataset
    assert dataset in ["MNIST", "FMNIST", 
        "KMNIST", "CIFAR10", "CIFAR100"], \
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
    # data format: batch_size(0) x channels(1) x height(2) x width(3)
    Dataset = {"MNIST":    datasets.MNIST,
               "KMNIST":   datasets.KMNIST,
               "FMNIST":   datasets.FashionMNIST,
               "CIFAR10":  datasets.CIFAR10, 
               "CIFAR100": datasets.CIFAR100}[dataset]

    # batch size (0)
    batch_size  = args.batch_size
    
    # image channels, height, width (1, 2, 3)
    if   dataset in ["MNIST", "KMNIST", "FMNIST"]:
        image_height, image_width, image_channels = 28, 28, 1
    elif dataset in ["CIFAR10", "CIFAR100"]:
        image_height, image_width, image_channels = 32, 32, 3

    output_classes = 100 if dataset == "CIFAR100" else 10

    image_mean = tuple([0.5] * image_channels)
    image_var  = tuple([0.5] * image_channels) 

    test_transform  = transforms.Compose(
        [transforms.Resize((image_height, image_width)),
         transforms.ToTensor(),
         transforms.Normalize(image_mean, image_var)])

    test_loader  =  torch.utils.data.DataLoader(Dataset(data_path,
        train = False, download = True, transform = test_transform),  
        batch_size = batch_size, shuffle = False, num_workers = num_workers)

    test_samples =  len(test_loader.dataset)
    print("# of test samples: ", test_samples)


    ## Models (Multi-layer Perceptron or Convolutional Neural Networks)

    # model architecture
    model_type = args.model_type
    assert model_type in ["MLP", "CNN", "CNN_", "VGGLike", "VGGLike_"], \
        "The type of model archiecture is not supported."

    # quantization levels
    quantization  = args.quantization
    assert quantization in ["binary", "ternary", "quanternary"], \
        "The type of weights quantization is not supported."
    nat_per_param = {"binary": math.log(2), "ternary": math.log(3),
        "quanternary": math.log(4)}[quantization]

    use_bias = args.use_bias

    print("Model: ", model_type, 
        "(Quantization: %s, Bias: %s)" % (quantization, use_bias))

    # multi-layer perceptron
    if model_type == "MLP":
        model = BayesNet(image_height = image_height, image_width = image_width,
            image_channels = image_channels, output_classes = output_classes,
            use_bias = use_bias, quantization = quantization)

    # convolutional networks
    else:
        pooling = args.pooling
        print("Pooling: ", pooling)

        Model = {"CNN":  BayesConvNet,  "VGGLike":  BayesVGGLike,
                 "CNN_": BayesConvNet_, "VGGLike_": BayesVGGLike_}[model_type]

        assert ((model_type in ["CNN",  "VGGLike"])  and (pooling in ["avg",  "max"])) or \
               ((model_type in ["CNN_", "VGGLike_"]) and (pooling in ["avg_", "prob_"])), \
            "The type of pooling layer is not supported."

        model = Model(image_height = image_height, image_width = image_width,
            image_channels = image_channels, output_classes = output_classes,
             use_bias = use_bias, quantization = quantization, pooling = pooling)

    # number of parameters in the neural network
    num_params_full, num_params_conv = model.num_params()
    num_params = num_params_conv + num_params_full
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

    test_nll_ppg, test_acc_ppg = 0., 0 # Probabilistic Propagation
    test_nll_map, test_acc_map = 0., 0 # Maximum A Posterior (MAP)
    test_nll_mcs, test_acc_mcs = 0., 0 # Monte-Carlo Sampling(MCS)

    with torch.no_grad():
        model.eval()

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # (1) Probabilistic Propagation
            (prob, loss_nll) = model(data, target, mode = "analytic")
            loss_nll = torch.mean(loss_nll) # for multi-gpu
            pred = prob.max(1, keepdim = True)[1]

            test_nll_ppg += loss_nll.item()
            test_acc_ppg += pred.eq(target.view_as(pred)).sum().item()

            # (2) Maximum A Posterior and point propagation
            (prob, loss_nll) = model(data, target, mode = "MAP")
            loss_nll = torch.mean(loss_nll) # for multi-gpu
            pred = prob.max(1, keepdim = True)[1]

            test_nll_map += loss_nll.item()
            test_acc_map += pred.eq(target.view_as(pred)).sum().item()

            # (3) Monte Carlo Sampling and point propagation
            for s in range(mc_samples):
                (prob, _) = model(data, target, mode = "sampling")
                # loss_nll = torch.mean(loss_nll) # multi-gpu

                prob_s = prob if s == 0 else prob_s + prob
                # loss_nll_s = loss_nll if s == 0 else loss_nll_s + loss_nll

            prob = prob_s / mc_samples
            pred = prob.max(1, keepdim = True)[1]

            # loss_nll = loss_nll_s / mc_samples
            loss_nll = F.nll_loss(torch.log(prob + 1e-6), target, reduction = "sum")

            test_nll_mcs += loss_nll.item()
            test_acc_mcs += pred.eq(target.view_as(pred)).sum().item()

    # (1) Probabilistic propagation (Analytical inference)
    test_nll_ppg /= test_samples
    print("Epoch {} (Analytical) , Avg. NLL: {:.4f}, Acc.: {}/{} ({:.2f}%)".format(
        eval_epoch, test_nll_ppg, test_acc_ppg, test_samples, 100. * test_acc_ppg / test_samples))
    test_acc_ppg =  100. * test_acc_ppg / test_samples

    # (2) Maximum A Posterior and point propagation
    test_nll_map /= test_samples
    print("Epoch {} (MAP-rounded), Avg. NLL: {:.4f}, Acc.: {}/{} ({:.2f}%)".format(
        eval_epoch, test_nll_map, test_acc_map, test_samples, 100. * test_acc_map / test_samples))
    test_acc_map =  100. * test_acc_map / test_samples

    # (3) Monte Carlo Sampling and point propagation
    test_nll_mcs /= test_samples
    print("Epoch {} (Monte-Carlo), Avg. NLL: {:.4f}, Acc.: {}/{} ({:.2f}%)".format(
        eval_epoch, test_nll_mcs, test_acc_mcs, test_samples, 100. * test_acc_mcs / test_samples))
    test_acc_mcs =  100. * test_acc_mcs / test_samples

    # save the statistics 
    np.savez(stats_file, 
        test_acc_ppg = test_acc_ppg, test_nll_ppg = test_nll_ppg,
        test_acc_map = test_acc_map, test_nll_map = test_nll_map,
        test_acc_mcs = test_acc_mcs, test_nll_mcs = test_nll_mcs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 
        "Testing Bayesian Quantized Networks (BQN).")

    ## Data format (Pytorch format)
    # batch size (0) x channels (1) x height (2) x width (3)

    # batch size (0)
    parser.add_argument("--batch-size",  default = 100,  type = int,
        help = "The batch size for testing.")

    ## Device (CPU, single GPU or multiple GPUs)
    
    # whether to use GPU for testing
    parser.add_argument('--use-cuda', dest = 'use_cuda', action = 'store_true',
        help = 'Use GPU for testing.')
    parser.add_argument('--no-cuda',  dest = 'use_cuda', action = 'store_false', 
        help = "Do not use GPU for testing.")
    parser.set_defaults(use_cuda = True)

    # whether to use multi-GPU for testing 
    parser.add_argument('--multi-gpu',  dest = 'multi_gpu', action = 'store_true',
        help = 'Use multiple GPUs for testing.')
    parser.add_argument('--single-gpu', dest = 'multi_gpu', action = 'store_false',
        help = 'Do not use multiple GPU for testing.')
    parser.set_defaults(multi_gpu = False)

    ## Models (MLP, CNN, VGGLike)
    parser.add_argument("--quantization", default = "binary", type = str, 
        help = "The type of quantization (options: binary/ternary/quanternary).")
    parser.add_argument("--model-type", default = "CNN", type = str,
        help = "The type of the model (options: MLP/CNN/VGGLike/CNN_/VGGLike_).")
    parser.add_argument("--pooling", default = "avg", type = str,
        help = "The type of pooling used in convolutional networks (options: avg/max/avg_/prob_).")

    parser.add_argument("--use-bias", dest = "use_bias", action = "store_true", 
        help = "Use bias in all layers of the model.")
    parser.add_argument("--no-bias",  dest = "use_bias", action = "store_false", 
        help = "Do not use bias in all layers of the model.")
    parser.set_defaults(use_bias = True)

    parser.add_argument("--model-name",  default = "default", type = str,
        help = "The model name (to create the associated folder).")
    parser.add_argument("--model-stamp", default = "default", type = str,
        help = "The time stamp of the model (as a suffix to the its name).")

    ## Paths (Data, Checkpoints, Results and TensorboardX)

    # inputs:  data
    parser.add_argument("--dataset", default = "MNIST", type = str,
        help = "The dataset used for training (options: MNIST/FMNIST/KMNIST/CIFAR10/CIFAR100).")
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
    parser.add_argument('--eval-epoch', default = 300, type = int, 
        help = 'Evaluate the model of specified epoch.')

    parser.add_argument("--mc-samples", default = 10, type = int, 
        help = "The number of Monte-Carlo samples to evaluate the model.")

    main(parser.parse_args())