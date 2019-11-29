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

    # number of worker for dataloaders 
    num_workers = min(num_gpus, 1) * 4

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

    # path to the folder of the tensorboardX file
    tensorboard_path = os.path.join(outputs_path, args.tensorboard_path)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    tensorboard_writer = SummaryWriter(tensorboard_path)


    ## Data formats and Dataloaders 
    # data format: batch_size(0) x channels(1) x height(2) x width(3)
    Dataset = {"MNIST":    datasets.MNIST,
               "KMNIST":   datasets.KMNIST,
               "FMNIST":   datasets.FashionMNIST,
               "CIFAR10":  datasets.CIFAR10, 
               "CIFAR100": datasets.CIFAR100}[dataset]

    output_classes = 100 if dataset == "CIFAR100" else 10

    # batch size and the log intervals (0)
    batch_size  = args.batch_size
    log_samples = args.log_samples
    assert log_samples % batch_size == 0, \
        "The argument log_samples should be a multiple of batch_size."

    # number of channels(1), image_height (2) and image_width (3)
    if args.default_parameters:
        if dataset in ["MNIST", "KMNIST", "FMNIST"]:
            image_height, image_width, image_channels = 28, 28, 1
            image_padding, image_flipping = 2, 0.0

        elif dataset in ["CIFAR10", "CIFAR100"]:
            image_height, image_width, image_channels = 32, 32, 3
            image_padding, image_flipping = 4, 0.5
    else: 
        image_height   = args.image_height
        image_width    = args.image_width
        image_channels = args.image_channels

        image_padding  = args.image_padding
        image_flipping = args.image_flipping

    image_mean = tuple([0.5] * image_channels)
    image_var  = tuple([0.5] * image_channels) 

    # preprocessing/transformation of the input images
    train_transform = transforms.Compose(
        [transforms.Resize((image_height, image_width)),
         transforms.RandomCrop((image_height, image_width), image_padding),
         transforms.RandomHorizontalFlip(image_flipping), 
         transforms.ToTensor(),
         transforms.Normalize(image_mean, image_var)])

    test_transform  = transforms.Compose(
        [transforms.Resize((image_height, image_width)),
         transforms.ToTensor(),
         transforms.Normalize(image_mean, image_var)])

    # dataloaders for training and test datasets
    train_loader = torch.utils.data.DataLoader(Dataset(data_path,
        train = True,  download = True, transform = train_transform), 
        batch_size = batch_size, shuffle = True,  num_workers = num_workers)

    train_samples = len(train_loader.dataset)
    print("# of training samples: ", train_samples)

    test_loader  = torch.utils.data.DataLoader(Dataset(data_path,
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


    ## Main script for learning and evaluation 
    epoch_num   = args.epoch_num
    save_epoch  = args.save_epoch

    # analytic inference (probability propagation) 
    test_acc_ppg_ = np.zeros(epoch_num, dtype = np.float)
    test_nll_ppg_ = np.zeros(epoch_num, dtype = np.float)

    # maximum a posterior  (point propagation)
    test_acc_map_ = np.zeros(epoch_num, dtype = np.float)
    test_nll_map_ = np.zeros(epoch_num, dtype = np.float)

    # monte-carlo sampling (point propagation)
    mc_samples = args.mc_samples # number of samples
    test_acc_mcs_ = np.zeros(epoch_num, dtype = np.float)
    test_nll_mcs_ = np.zeros(epoch_num, dtype = np.float)

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

    # 
    for epoch in range(start_epoch, epoch_num):
        learning_rate = optimizer.param_groups[0]['lr']
        tensorboard_writer.add_scalar('lr', learning_rate, epoch + 1)
        print("Epoch %d, Learning rate: %2f" % (epoch + 1, learning_rate))

        ## Phase 1: Learning on training set
        model.train()

        samples, train_acc, train_nll = 0, 0, 0.

        # initialize the statistics
        LOSS, NLL, ACC = 0., 0., 0

        for (data, target) in train_loader:
            data, target = data.to(device), target.to(device)
            total_samples += batch_size
            samples += batch_size

            # predict the outputs with probabilistic propagation
            prob, loss_nll = model(data, target, mode = "analytic")
            pred = prob.max(1, keepdim = True)[1]
            loss_nll = torch.mean(loss_nll) # for multi-gpu
            correct = pred.eq(target.view_as(pred)).sum().item()

            # compute the regularizer based on the (joint) entropy
            entropy_full, entropy_conv =  model.sum_entropy(device) if \
                not multi_gpu else model.module.sum_entropy(device)
            entropy  = entropy_full + entropy_conv

            loss_reg = - (args.warm_lamb if epoch < args.warm_epoch
                else args.lamb) * entropy * batch_size / train_samples

            loss = loss_nll + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate the statistics
            LOSS += loss.item()
            NLL  += loss_nll.item()
            ACC  += correct

            if samples % args.log_samples == 0:
                entropy_full_norm = entropy_full.item() / (nat_per_param * num_params_full + 1e-6)
                entropy_conv_norm = entropy_conv.item() / (nat_per_param * num_params_conv + 1e-6)

                tensorboard_writer.add_scalar('entropy_full', entropy_full_norm, total_samples)
                tensorboard_writer.add_scalar('entropy_conv', entropy_conv_norm, total_samples)

                print("Epoch: {} [{}/{} ({:.1f}%)], Total loss: {:.4f}, Avg. NLL: {:.4f}, Acc: {}/{}, \
                    Conv. ent. (x 1e-3): {:.4f}, Full. ent. (x 1e-3): {:.4f}".format(
                    epoch + 1, samples, train_samples, 100. * samples / train_samples, # progress information
                    LOSS / args.log_samples, NLL / args.log_samples, ACC, args.log_samples,
                    1e3 * entropy_conv_norm, 1e3 * entropy_full_norm))

                train_nll += NLL
                train_acc += ACC

                tensorboard_writer.add_scalar('train_loss', LOSS / args.log_samples, total_samples)
                tensorboard_writer.add_scalar('train_nll',   NLL / args.log_samples, total_samples)
                tensorboard_writer.add_scalar('train_acc',   ACC / args.log_samples, total_samples)

                # re-initialize the statistics
                LOSS, NLL, ACC = 0., 0., 0

        train_nll /= train_samples
        print("Epoch: {} (Training), Avg. NLL: {:.4f}, Acc.: {}/{} ({:.2f}%)".format(
            epoch + 1, train_nll, train_acc, train_samples, 100. * train_acc / train_samples))
        train_acc =  100. * train_acc / train_samples

        # break the training process if numerical error 
        if math.isnan(train_nll) or math.isinf(train_nll): break


        ## Phase 2: Evaluation on the validation set
        model.eval()

        test_nll_ppg, test_acc_ppg = 0., 0 # Probabilistic Propagation
        test_nll_map, test_acc_map = 0., 0 # Maximum A Posterior (MAP)
        test_nll_mcs, test_acc_mcs = 0., 0 # Monte-Carlo Sampling(MCS)

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                # (1) Probabilistic Propagation
                (prob, loss_nll) = model(data, target, mode = "analytic")
                pred = prob.max(1, keepdim = True)[1]
                loss_nll = torch.mean(loss_nll) # for multi-gpu

                test_nll_ppg += loss_nll.item()
                test_acc_ppg += pred.eq(target.view_as(pred)).sum().item()

                # (2) Maximum A Posterior and point propagation
                (prob, loss_nll) = model(data, target, mode = "MAP")
                pred = prob.max(1, keepdim = True)[1]
                loss_nll = torch.mean(loss_nll) # for multi-gpu

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


        ## Phase 3: Logging the learning curves and checkpoints
        if args.rate_decay: scheduler.step()

        # (1) Probabilistic propagation (Analytical inference)
        test_nll_ppg /= test_samples
        print("Epoch {} (Analytical) , Avg. NLL: {:.4f}, Acc.: {}/{} ({:.2f}%)".format(
            epoch + 1, test_nll_ppg, test_acc_ppg, test_samples, 100. * test_acc_ppg / test_samples))
        test_acc_ppg =  100. * test_acc_ppg / test_samples

        test_acc_ppg_[epoch] = test_acc_ppg 
        test_nll_ppg_[epoch] = test_nll_ppg

        tensorboard_writer.add_scalar('test_acc_ppg', test_acc_ppg, epoch + 1)
        tensorboard_writer.add_scalar('test_nll_ppg', test_nll_ppg, epoch + 1)

        # (2) Maximum A Posterior and point propagation
        test_nll_map /= test_samples
        print("Epoch {} (MAP-rounded), Avg. NLL: {:.4f}, Acc.: {}/{} ({:.2f}%)".format(
            epoch + 1, test_nll_map, test_acc_map, test_samples, 100. * test_acc_map / test_samples))
        test_acc_map =  100. * test_acc_map / test_samples

        test_acc_map_[epoch] = test_acc_map
        test_nll_map_[epoch] = test_nll_map

        tensorboard_writer.add_scalar('test_acc_map', test_acc_map, epoch + 1)
        tensorboard_writer.add_scalar('test_nll_map', test_nll_map, epoch + 1)

        # (3) Monte Carlo Sampling and point propagation
        test_nll_mcs /= test_samples
        print("Epoch {} (Monte-Carlo), Avg. NLL: {:.4f}, Acc.: {}/{} ({:.2f}%)".format(
            epoch + 1, test_nll_mcs, test_acc_mcs, test_samples, 100. * test_acc_mcs / test_samples))
        test_acc_mcs =  100. * test_acc_mcs / test_samples

        test_acc_mcs_[epoch] = test_acc_mcs
        test_nll_mcs_[epoch] = test_nll_mcs

        tensorboard_writer.add_scalar('test_acc_mcs', test_acc_mcs, epoch + 1)
        tensorboard_writer.add_scalar('test_nll_mcs', test_nll_mcs, epoch + 1)

        # update the best model so far
        if test_nll_ppg < min_test_nll:
            min_epoch, min_test_nll = epoch + 1, test_nll_ppg

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
    np.savez(stats_file, 
        test_acc_ppg = test_acc_ppg_, test_nll_ppg = test_nll_ppg_,
        test_acc_map = test_acc_map_, test_nll_map = test_nll_map_,
        test_acc_mcs = test_acc_mcs_, test_nll_mcs = test_nll_mcs_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 
        "Training Bayesian Quantized Networks (BQN).")

    ## 1) Data format (Pytorch format)
    # batch size (0) x channels (1) x height (2) x width (3)

    # dataset name
    parser.add_argument("--dataset", default = "MNIST", type = str,
        help = "The dataset used for training (options: MNIST/FMNIST/KMNIST/CIFAR10/CIFAR100).")

    parser.add_argument('--default-parameters',   dest = 'default_parameters', action = 'store_true',
        help = 'Use  default  parameters (format/augmentation) for the given dataset.')
    parser.add_argument('--specified-parameters', dest = 'default_parameters', action = 'store_false',
        help = 'Use specified parameters (format/augmentation) for the given dataset.')
    parser.set_defaults(default_parameters = True)

    # image format: channels (1), height (2), width (3)
    parser.add_argument("--image-height", default =  32, type = int,
        help = "The image height of each sample.")
    parser.add_argument("--image-width",  default =  32, type = int,
        help = "The image width  of each sample.")
    parser.add_argument("--image-channels", default = 3, type = int,
        help = "The number of channels in each sample.")
    
    # data augmentation (in learning phase)
    parser.add_argument("--image-padding",  default =  2,  type = int,
        help = "The number of padded pixels along height/width.")
    parser.add_argument("--image-flipping", default = 0.5, type = float, 
        help = "The probability of horizontal filpping of the images")

    ## 2) Paths (Data, Checkpoints, Results and TensorboardX)

    # inputs:  data
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
    parser.add_argument('--tensorboard-path', default = 'tensorboard', type = str,
        help = 'The folder for the tensorboardX files.')

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

    ## 4) Models (MLP, CNN, VGGLike)
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

    ## 5) Hyperparameters for learning
    parser.add_argument("--epoch-num",  default = 300, type = int,
        help = "The total number of epochs for training.")
    parser.add_argument("--save-epoch", default =  20, type = int,
        help = "The interval of epochs to save a checkpoint.")

    parser.add_argument("--batch-size",  default =  100, type = int,
        help = "The batch size for training.")
    parser.add_argument("--log-samples", default = 5000, type = int, 
        help = "Log the learning curve every log_samples.")

    parser.add_argument("--lamb", default = 1e-3, type = float,
        help = "The coefficient of the regularizer after warm-up.")

    parser.add_argument("--warm-epoch", default =  5,  type = int,
        help = "The number of epochs for training in warm-up mode.")
    parser.add_argument("--warm-lamb", default = -1e-4, type = float, 
        help = "The coefficient of the regularizer in warm-up mode.")

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
    parser.add_argument("--decay-rate", default = 0.98, type = float,
        help = "The learning rate is decayed by decay_rate every decay_epoch.")

    # evaluation
    parser.add_argument("--mc-samples", default = 5, type = int, 
        help = "The number of Monte-Carlo samples to evaluate the model.")

    main(parser.parse_args())