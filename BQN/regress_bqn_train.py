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

    # fix the random seed to reproductivity (if --use-seed) 
    if not args.use_seed:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        if use_cuda: 
            torch.cuda.manual_seed_all(args.random_seed)


    ## Paths (Dataset, Checkpoints, Statistics and TensorboardX)
    
    # path to the folder of all datasets
    data_path = args.data_path
    if not os.path.exists(data_path): 
        os.makedirs(data_path)

    # path to the folder of specified dataset
    dataset = args.dataset
    assert dataset in ["Boston"], "The specified dataset is not supported."
    print("Dataset: ", dataset)

    if args.dataset_path == "default":
        dataset_path = dataset
    else: # if args.dataset_path != "default":
        dataset_path = args.dataset_path

    data_path = os.path.join(data_path, dataset_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # path to the folder of all outputs (for the dataset)
    outputs_path = args.outputs_path
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)

    outputs_path = os.path.join(outputs_path, dataset_path)
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)

    # create the name of the current network architecture
    if args.network_name == "default":
        model_param = str(args.lamb)
        network_name = args.model_type + '_' + model_param
    else: # if args.model_name != "default":
        network_name = args.network_name

    outputs_path = os.path.join(outputs_path, network_name)
    if not os.path.exists(outputs_path): 
        os.makedirs(outputs_path)

    # create the name (and time stamp) of the current model
    if args.model_name  == "default":
        model_name = "S" + str(args.random_seed)
    else: # if args.model_name != "default":
        model_name = args.model_name

    if args.model_stamp == "default":
        model_stamp = datetime.datetime.now().strftime("%m%d")
    else: # if args.model_stamp != "default":
        model_stamp = args.model_stamp

    model_name += '_' + model_stamp

    outputs_path = os.path.join(outputs_path, model_name)
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)

    # path to the folder of checkpoints
    model_path = os.path.join(outputs_path, args.model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # path to the folder/file of the evaluation statistics
    stats_path = os.path.join(outputs_path, args.stats_path)
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)

    stats_file = os.path.join(stats_path, args.stats_file)


    ## Data formats and Dataloaders 
    # data format: batch_size(0) x intut_length (1)

    # batch size and the log intervals (0)
    batch_size  = args.batch_size
    log_samples = args.log_samples
    assert log_samples % batch_size == 0, \
        "The argument log_samples should be a multiple of batch_size."

    if dataset == "Boston":
        train_dataset = Dataset_boston_housing('../data/Boston/boston_housing_nor_train.pkl')
        test_dataset  = Dataset_boston_housing('../data/Boston/boston_housing_nor_val.pkl')
    else: 
        raise NotImplementedError

    # number of worker for dataloaders 
    num_workers = min(num_gpus, 1) * 4

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
        batch_size = batch_size, shuffle = True,  num_workers = num_workers, pin_memory = False)

    test_loader  = torch.utils.data.DataLoader(dataset = test_dataset,
        batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = False)

    train_samples = len(train_loader.dataset)
    print("# of training samples: ",   train_samples)

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


    ## Main script for learning and evaluation 
    epoch_num   = args.epoch_num
    save_epoch  = args.save_epoch

    # analytic inference (probability propagation)
    test_rmse_ = np.zeros(epoch_num, dtype = np.float)
    test_nll_  = np.zeros(epoch_num, dtype = np.float)

    # maximum a posterior  (point propagation)
    test_rmse_map_ = np.zeros(epoch_num, dtype = np.float)
    test_nll_map_  = np.zeros(epoch_num, dtype = np.float)

    # monte-carlo sampling (point propagation)
    mc_samples = args.mc_samples
    test_rmse_mcs_ = np.zeros(epoch_num, dtype = np.float)
    test_nll_mcs_  = np.zeros(epoch_num, dtype = np.float)

    # initial learning rate
    learning_rate = args.learning_rate

    # recover the model to resume training (if required)
    if args.start_begin:
        model_file = None
        start_epoch, total_samples = 0, 0
        min_epoch, min_test_nll = 0, float("inf")

    else: # if args.start_exist:
        model_file = os.path.join(model_path, 'training_last.pt' 
            if args.start_last else "training_%d.pt" % args.start_epoch)
        print(model_file)
        assert os.path.exists(model_file), \
            "The specified model is not found in the folder."
        
    if model_file is not None:
        checkpoint  = torch.load(model_file)

        # model parameters
        model.load_state_dict(checkpoint["model_state_dict"])

        # training progress
        start_epoch   = checkpoint["epoch"]
        total_samples = checkpoint["total_samples"]

        # best model and its negative likelihood
        min_epoch = checkpoint["min_epoch"]
        min_test_nll = checkpoint["min_test_nll"]

        # learning rate
        learning_rate = checkpoint["learning_rate"]

    # optimizer and corresponding scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
        step_size = args.decay_epoch, gamma = args.decay_rate)

    # test statistics
    test_rmse_ = np.zeros(epoch_num, dtype = np.float)
    test_nll_  = np.zeros(epoch_num, dtype = np.float)

    test_rmse_map_ = np.zeros(epoch_num, dtype = np.float)
    test_nll_map_  = np.zeros(epoch_num, dtype = np.float)

    mc_samples = args.mc_samples
    test_rmse_mcs_ = np.zeros(epoch_num, dtype = np.float)
    test_nll_mcs_  = np.zeros(epoch_num, dtype = np.float)

    total_samples = 0
    for epoch in range(epoch_num):
        learning_rate = optimizer.param_groups[0]['lr']
        print("Epoch %d, Learning rate: %2f" % (epoch + 1, learning_rate))

        ## Phase 1: Learning on training set
        model.train()

        samples = 0
        train_nll, train_rmse = 0., 0.

        # initialize the statistics
        LOSS, NLL, RMSE = 0., 0., 0.

        for (data, target) in train_loader:
            data, target = data.to(device), target.to(device)
            total_samples += batch_size
            samples += batch_size

            # predict the outputs with probabilistic propagation
            outputs, loss_nll = model(data, target, mode = "analytic")
            loss_nll = torch.mean(loss_nll) # for multi-gpu

            rmse = torch.sum((outputs[0] - target) ** 2, dim = 1) ** 0.5

            # compute the regularizer based on the (joint) entropy
            (entropy, ) =  model.sum_entropy(device) if \
                not multi_gpu else model.module.sum_entropy(device)

            loss_reg = - args.lamb * entropy * batch_size / train_samples
            loss = loss_nll + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate the statistics
            LOSS += loss.item()
            NLL  += loss_nll.item()
            RMSE += torch.sum(rmse).item()

            if samples % args.log_samples == 0:

                print("Epoch: {} [{}/{} ({:.1f}%)], Total loss: {:.4f}, Avg. NLL: {:.4f}, \
                    Avg. RMSE: {:4f}, Ent. (x 1e-3): {:.4f}".format(
                    epoch + 1, samples, train_samples, 100. * samples / train_samples, # progress information
                    LOSS / args.log_samples, NLL / args.log_samples, # loss functions
                    RMSE / args.log_samples, 1e3 * entropy.item() / (nat_per_param * num_params)))

                train_nll  += NLL
                train_rmse += RMSE

                # re-initialize the statistics
                LOSS, NLL, RMSE = 0., 0., 0

        train_nll  /= train_samples
        train_rmse /= train_samples 
        print("Epoch: {} (Training), Avg. NLL: {:.4f}, Avg RMSE: {:4f}".format(
            epoch + 1, train_nll, train_rmse))

        if math.isnan(train_nll) or math.isinf(train_nll): break


        ## Phase 2: Evaluation on the validation set
        model.eval()

        test_nll, test_rmse = 0., 0. # Probabilistic Propagation
        test_nll_map, test_rmse_map = 0., 0.
        test_nll_mcs, test_rmse_mcs = 0., 0.

        with torch.no_grad():
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


        ## Phase 3: Logging the learning curves and checkpoints
        if args.rate_decay: scheduler.step()

        # (1) Probabilistic propagation (Analytical inference)
        test_nll  /= test_samples
        test_rmse /= test_samples 
        print("Epoch {} (Analytical) , Avg. NLL: {:.4f}, Avg. RMSE: {:4f}".format(
            epoch + 1, test_nll, test_rmse))

        test_nll_[epoch]  = test_nll
        test_rmse_[epoch] = test_rmse

        # (2) Maximum A Posterior and point propagation
        test_nll_map  /= test_samples
        test_rmse_map /= test_samples 
        print("Epoch {} (Analytical) , Avg. NLL: {:.4f}, Avg. RMSE: {:4f}".format(
            epoch + 1, test_nll_map, test_rmse_map))

        test_nll_map_[epoch]  = test_nll_map
        test_rmse_map_[epoch] = test_rmse_map

        # (3) Monte Carlo Sampling and point propagation
        test_nll_mcs  /= test_samples
        test_rmse_mcs /= test_samples 
        print("Epoch {} (Analytical) , Avg. NLL: {:.4f}, Avg. RMSE: {:4f}".format(
            epoch + 1, test_nll_mcs, test_rmse_mcs))

        test_nll_mcs_[epoch]  = test_nll_mcs
        test_rmse_mcs_[epoch] = test_rmse_mcs

        # update the best model so far
        if test_nll < min_test_nll:
            min_epoch, min_test_nll = epoch + 1, test_nll

        # save the currrent model as a checkpoint
        checkpoint_info = {
            "epoch": epoch + 1, "total_samples": total_samples, # training progress 
            "min_epoch": min_epoch, "min_test_nll": min_test_nll, # best model and loss 
            "learning_rate": optimizer.param_groups[0]['lr'], # current learning rate 
            "model_state_dict": model.state_dict() # model parameters
        }

        torch.save(checkpoint_info, os.path.join(model_path, 'training_last.pt'))

        if (epoch + 1) % save_epoch == 0:
            torch.save(checkpoint_info, os.path.join(model_path, 'training_%d.pt' % (epoch + 1)))

        if (epoch + 1) == min_epoch:
            torch.save(checkpoint_info, os.path.join(model_path, 'training_best.pt'))


    # save the statistics 
    np.savez(stats_file, test_rmse = test_rmse_, test_nll = test_nll_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 
        "Training Bayesian Quantized Networks (BQN).")

    ## 1) Data format (Pytorch format)
    # batch size (0) x input_length (1)

    # batch size and log interval (0)
    parser.add_argument("--log-samples", default = 100, type = int, 
        help = "Log the learning curve every log_samples.")
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
        help = "The folder for the training and test datasets.")

    # outputs: checkpoints, statistics and tensorboard
    parser.add_argument("--outputs-path",  default = "../outputs_BQN", type = str,
        help = "The path to the folder storing outputs from training.")
    parser.add_argument("--network-name", default = "default", type = str, 
        help = "The architecture model (to create the folder).")

    parser.add_argument("--model-name",  default = "default", type = str,
        help = "The model name (to create the folder).")
    parser.add_argument("--model-stamp", default = "default", type = str,
        help = "The time stamp of the model (as a suffix to the its name).")

    parser.add_argument("--model-path", default = "models", type = str,
        help = "The folder for all checkpoints in training.")
    parser.add_argument("--stats-path", default = "stats",  type = str,
        help = "The folder for the evaluation statistics.")
    parser.add_argument("--stats-file", default = "curve",  type = str, 
        help = "The file name for the learning curve.")

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

    # random seed for reproducibility
    parser.add_argument('--use-seed', dest = 'use_seed', action = 'store_true', 
        help = 'Fix the random seed to reproduce the model.')
    parser.add_argument('--no-seed',  dest = 'use_seed', action = 'store_false', 
        help = 'Randomly choose the random seed.')
    parser.set_defaults(use_seed = True)

    parser.add_argument('--random-seed', default = 0, type = int, 
        help = 'The random seed number (to reproduce the model).')

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

    ## 5) Hyperparameters for learning
    parser.add_argument("--epoch-num", default = 400, type = int,
        help = "The total number of epochs for training.")
    parser.add_argument("--save-epoch", default = 50, type = int,
        help = "The interval of epochs to save a checkpoint.")

    parser.add_argument("--lamb", default = 1e-3, type = float,
        help = "The coefficient of the regularizer.")

    # whether to start training from 
    parser.add_argument('--start-begin', dest = 'start_begin', action = 'store_true', 
        help = 'Start training a new model from the beginning.')
    parser.add_argument('--start-exist', dest = 'start_begin', action = 'store_false',
        help = 'Resume training from an existing model.')
    parser.set_defaults(start_begin = True)

    # if start_begin is False (--start-exist)
    parser.add_argument('--start-last', dest = 'start_last', action = 'store_true', 
        help = 'Resume training from the last available model.')
    parser.add_argument('--start-spec', dest = 'start_last', action = 'store_false', 
        help = 'Resume training from the model of the specified epoch.')
    parser.set_defaults(start_last = True)

    # if start_last is False (--start-spec)
    parser.add_argument('--start-epoch', default = 0, type = int, 
        help = 'The number of epoch to resume training.')

    # learning rate scheduling
    parser.add_argument("--learning-rate", default = 1e-2, type = float,
        help = "Initial learning rate of the ADAM optimizer.")

    parser.add_argument("--learning-rate-decay", dest = "rate_decay", action = 'store_true',
        help = "Learning rate is decayed during training.")
    parser.add_argument("--learning-rate-fixed", dest = "rate_decay", action = 'store_false', 
        help = "Learning rate is fixed during training.")
    parser.set_defaults(rate_decay = True)

    # if rate_decay is True (--learning-rate-decay)
    parser.add_argument("--decay-epoch", default = 1, type = int,
        help = "The learning rate is decayed by decay_rate every decay_epoch.")
    parser.add_argument("--decay-rate", default = 0.99, type = float,
        help = "The learning rate is decayed by decay_rate every decay_epoch.")

    # evaluation
    parser.add_argument("--mc-samples", default = 10, type = int, 
        help = "The number of Monte-Carlo samples to evaluate the model.")

    main(parser.parse_args())