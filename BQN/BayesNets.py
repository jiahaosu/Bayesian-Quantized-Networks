import torch
import torch.nn as nn
import torch.nn.functional as F

# linear layers
from layers.linear import BayesLinear, BayesConv2d
# activation and output layers
from layers.activation import BayesSign, BayesSoftmax, BayesGaussian 
# pooling layers
from layers.pooling import BayesAvgPool2d,  BayesMaxPool2d
from layers.pooling import BayesAvgPool2d_, BayesProbPool2d_
# normalization layers
from layers.normalization import BayesBatchNorm1d, BayesBatchNorm2d


## Bayesian Quantized Regressive Network (BQ-ReNet)
# 2 fully-connected layers
class BayesReNet(nn.Module):
    def __init__(self, input_length = 13, output_length = 1,
        use_bias = True, quantization = "binary"):
        """
        Initialization of a Bayesian Quantized Regressive Network (BQ-ReNet).
        """
        super(BayesReNet, self).__init__()
        self.layers = nn.ModuleDict()

        # Layer 1: Fully connected layer + Sign function
        self.layers["full"]  = BayesLinear(input_length, 50, use_bias = use_bias, 
            quantization = quantization, input_type = "dirac")
        self.layers["sign"]  = BayesSign()

        # Layer 2: Fully connected layer + Guassian function
        self.layers["gauss"] = BayesGaussian(50, output_length, scale = 1)

        # names of the fully-connected and convolutional layers
        self.full_layers = ["full"]

    def num_params(self):
        """
        Compute the total number of parameters in the BQ-ReNet.
        """
        num_params_full = 0
        for l in self.full_layers:
            num_params_full += self.layers[l].num_params

        return (num_params_full, )

    def sum_entropy(self, device):
        """
        Compute the joint entropy of the BQ-ReNet.
        """
        sum_entropy_full = torch.tensor(0., 
            dtype = torch.float32, device = device)
        for l in self.full_layers:
            sum_entropy_full += self.layers[l].entropy()

        return (sum_entropy_full, )
            
    def forward(self, inputs, targets, mode = "analytic", reduction = "sum"):
        """
        Computation of the Bayesian Quantized Regressive Network (BQ-ReNet).
        """

        inputs = inputs.view(inputs.size(0), -1) 

        # Layer 1: Fully-connected layer + Sign function
        inputs = self.layers["full"](inputs, mode = mode)
        inputs = self.layers["sign"](inputs, mode = mode)

        # Layer 2: Fully connected layer + Guassian function
        (outputs, nll) = self.layers["gauss"](inputs, 
            targets, mode = mode, return_nll = reduction)

        return (outputs, nll.reshape([1]))


## Bayesian Quantized Multi-Layer Perceptron (BQ-MLP) 
# -- 3 fully-connected layers
class BayesNet(nn.Module):
    def __init__(self, 
        image_channels = 1, image_height = 28, image_width = 28,
        output_classes = 10, use_bias = True, quantization = "binary"):
        """
        Initialization of a Bayesian Quantized Multi-Layer Perceptron (BQ-MLP).
        """
        super(BayesNet, self).__init__()
        self.layers = nn.ModuleDict()

        # Layer 1: Fully connected layer + Sign function
        input_length = image_channels * image_height * image_width
        self.layers["fc1"] = BayesLinear(input_length, 512, use_bias = use_bias, 
            quantization = quantization, input_type = "dirac")
        self.layers["nl1"] = BayesSign()

        # Layer 2: Fully connected layer + Sign function
        self.layers["fc2"] = BayesLinear(512, 256, use_bias = use_bias,
            quantization = quantization, input_type = "discrete")
        self.layers["nl2"] = BayesSign()

        # Layer 3: Fully connected layer + Softmax function
        self.layers["fc3"] = BayesLinear(256, output_classes, use_bias = use_bias, 
            quantization = quantization, input_type = "discrete")
        self.layers["softmax"] = BayesSoftmax(units = 256, scale = 1)

        # names of the fully-connected and convolutional layers
        self.full_layers = ["fc1", "fc2", "fc3"]
        self.conv_layers = []

    def num_params(self):
        """
        Compute the total number of parameters in the BQ-MLP.
        """
        num_params_full = 0
        for l in self.full_layers:
            num_params_full += self.layers[l].num_params

        return (num_params_full, 0)

    def sum_entropy(self, device):
        """
        Compute the joint entropy of the BQ-MLP.
        """
        sum_entropy_full = torch.tensor(0., 
            dtype = torch.float32, device = device)
        for l in self.full_layers:
            sum_entropy_full += self.layers[l].entropy()

        return (sum_entropy_full, torch.tensor(0., 
            dtype = torch.float32, device = device))
            
    def forward(self, inputs, targets, mode = "analytic", reduction = "sum"):
        """
        Computation of the Bayesian Quantized Multi-Layer Perceptron (BQ-MLP).
        """

        inputs = inputs.view(inputs.size(0), -1) 

        # Layer 1: Fully-connected layer + Sign function
        inputs = self.layers["fc1"](inputs, mode = mode)
        inputs = self.layers["nl1"](inputs, mode = mode)

        # Layer 2: Fully-connected layer + Sign function
        inputs = self.layers["fc2"](inputs, mode = mode)
        inputs = self.layers["nl2"](inputs, mode = mode)

        # Layer 3: Fully-connected layer + Sign function
        inputs = self.layers["fc3"](inputs, mode = mode)
        (prob, nll) = self.layers["softmax"](inputs, 
            targets, mode = mode, return_nll = reduction)

        return (prob, nll.reshape([1]))


## Bayesian Quantized Convolutional Neural Network (BQ-CNN)
# 2 convolutional layers + 2 fully-connected layers 
class BayesConvNet(nn.Module):
    def __init__(self, image_channels = 1, image_height = 28, image_width = 28,
        output_classes = 10, use_bias = True, quantization = "binary", pooling = "max"):
        """
        Initialization of Bayesian Quantized Convolutional Neural Networks (BQ-CNN).
        """
        super(BayesConvNet, self).__init__()
        self.layers = nn.ModuleDict()

        # type of pooling layers
        assert pooling in ["avg", "max"], \
            "The type of pooling layer is not supported."

        BayesPool2d = {"avg": BayesAvgPool2d, 
                       "max": BayesMaxPool2d}[pooling]

        # Layer 1: 5 x 5 Convolution layer + 2 x 2 Pooling 
        # + Batch normalization + Sign activation 
        self.layers["conv1"] = BayesConv2d(image_channels, 64, 
            kernel_size = 5, stride = 1, padding = 2, use_bias = use_bias, 
            quantization = quantization, input_type = "dirac")
        self.layers["pool1"] = BayesPool2d(kernel_size = 2)
        self.layers["norm1"] = BayesBatchNorm2d(64)
        self.layers["sign1"] = BayesSign()
        
        # Layer 2: 5 x 5 Convolution Layer + 2 x 2 Pooling 
        # + Batch normalization + Sign activation 
        self.layers["conv2"] = BayesConv2d(64, 64, 
            kernel_size = 5, stride = 1, padding = 2, use_bias = use_bias, 
            quantization = quantization, input_type = "discrete")
        self.layers["pool2"] = BayesPool2d(kernel_size = 2)
        self.layers["norm2"] = BayesBatchNorm2d(64)
        self.layers["sign2"] = BayesSign()
        
        # Flatten the feature maps into a vector
        input_length = (image_height // 4) * (image_width // 4) * 64

        # Layer 3: Fully connected layer + Batch normalization + Sign activation
        self.layers["full3"] = BayesLinear(input_length, 1024, 
            use_bias = use_bias, quantization = quantization, input_type = "discrete")
        self.layers["norm3"] = BayesBatchNorm1d(1024)
        self.layers["sign3"] = BayesSign()

        # Layer 4: Fully connected layer + Softmax output layer
        self.layers["full4"] = BayesLinear(1024, output_classes, 
            use_bias = use_bias, quantization = quantization, input_type = "discrete")
        self.layers["softmax"] = BayesSoftmax(units = 1024, scale = 1)

        # names of the fully-connected and convolutional layers
        self.full_layers = ["full3", "full4"]
        self.conv_layers = ["conv1", "conv2"]

    def num_params(self):
        """
        Compute the total number of parameters in the BQ-CNN.
        """
        num_params_full = 0
        for l in self.full_layers:
            num_params_full += self.layers[l].num_params

        num_params_conv = 0
        for l in self.conv_layers:
            num_params_conv += self.layers[l].num_params

        return (num_params_full, num_params_conv)

    def sum_entropy(self, device):
        """
        Compute the joint entropy of the BQ-CNN.
        """
        sum_entropy_full = torch.tensor(0., 
            dtype = torch.float32, device = device)
        for l in self.full_layers:
            sum_entropy_full += self.layers[l].entropy()

        sum_entropy_conv = torch.tensor(0., 
            dtype = torch.float32, device = device)
        for l in self.conv_layers:
            sum_entropy_conv += self.layers[l].entropy()

        return (sum_entropy_full, sum_entropy_conv)

    def forward(self, inputs, targets, mode = "analytic", reduction = "sum"):
        """
        Computation of Bayesian Quantized Convolutional Neural Network (BQ-CNN).
        """

        # Layer 1: 5 x 5 Convolution layer + 2 x 2 Pooling 
        # + Batch normalization + Sign activation
        inputs = self.layers["conv1"](inputs, mode = mode)
        inputs = self.layers["pool1"](inputs, mode = mode)
        inputs = self.layers["norm1"](inputs, mode = mode)
        inputs = self.layers["sign1"](inputs, mode = mode)

        # Layer 2: 5 x 5 Convolution Layer + 2 x 2 Pooling 
        # + Batch normalization + Sign activation
        inputs = self.layers["conv2"](inputs, mode = mode)
        inputs = self.layers["pool2"](inputs, mode = mode)
        inputs = self.layers["norm2"](inputs, mode = mode)
        inputs = self.layers["sign2"](inputs, mode = mode)

        # Flatten the feature maps into a vector
        inputs = inputs.view(inputs.size(0), -1)

        # Layer 3: Fully connected layer + Batch normalization + Sign activation
        inputs = self.layers["full3"](inputs, mode = mode)
        inputs = self.layers["norm3"](inputs, mode = mode)
        inputs = self.layers["sign3"](inputs, mode = mode)

        # Layer 4: Fully connected layer + Softmax output layer
        inputs = self.layers["full4"](inputs, mode = mode)
        prob, nll = self.layers["softmax"](inputs, 
            targets, mode = mode, return_nll = reduction)

        return (prob, nll.reshape([1]))


## Bayesian Quantized Convolutional Neural Network (BQ-CNN)
# 2 convolutional layers + 2 fully-connected layers 
class BayesConvNet_(nn.Module):
    def __init__(self, image_channels = 1, image_height = 28, image_width = 28,
        output_classes = 10, use_bias = True, quantization = "binary", pooling = "max"):
        """
        Initialization of Bayesian Quantized Convolutional Neural Networks (BQ-CNN).
        """
        super(BayesConvNet_, self).__init__()
        self.layers = nn.ModuleDict()

        # type of pooling layers (and distribution of activation)
        assert pooling in ["prob_", "avg_"], \
            "The type of pooling layer is not supported."

        BayesPool2d = {"avg_":  BayesAvgPool2d_, 
                       "prob_": BayesProbPool2d_}[pooling]

        activation_type = {"avg_":  'gaussian', 
                           "prob_": 'discrete'}[pooling]

        # Layer 1: 5 x 5 Convolution layer + Batch normalization 
        # + Sign activation + 2 x 2 Pooling 
        self.layers["conv1"] = BayesConv2d(image_channels, 64, 
            kernel_size = 5, stride = 1, padding = 2, use_bias = use_bias, 
            quantization = quantization, input_type = "dirac")
        self.layers["norm1"] = BayesBatchNorm2d(64)
        self.layers["sign1"] = BayesSign()
        self.layers["pool1"] = BayesPool2d(kernel_size = 2)
        
        # Layer 2: 5 x 5 Convolution Layer + Batch normalization
        # + Sign activation + 2 x 2 Pooling
        self.layers["conv2"] = BayesConv2d(64, 64, 
            kernel_size = 5, stride = 1, padding = 2, use_bias = use_bias, 
            quantization = quantization, input_type = activation_type)
        self.layers["norm2"] = BayesBatchNorm2d(64)
        self.layers["sign2"] = BayesSign()
        self.layers["pool2"] = BayesPool2d(kernel_size = 2)
        
        # Flatten the feature maps into a vector
        input_length = (image_height // 4) * (image_width // 4) * 64

        # Layer 3: Fully connected layer + Batch normalization + Sign activation
        self.layers["full3"] = BayesLinear(input_length, 1024, 
            use_bias = use_bias, quantization = quantization, input_type = activation_type)
        self.layers["norm3"] = BayesBatchNorm1d(1024)
        self.layers["sign3"] = BayesSign()

        # Layer 4: Fully connected layer + Softmax output layer
        self.layers["full4"] = BayesLinear(1024, output_classes, 
            use_bias = use_bias, quantization = quantization, input_type = "discrete")
        self.layers["softmax"] = BayesSoftmax(units = 1024, scale = 1)

        # names of the fully-connected and convolutional layers
        self.full_layers = ["full3", "full4"]
        self.conv_layers = ["conv1", "conv2"]

    def num_params(self):
        """
        Compute the total number of parameters in the BQ-CNN.
        """
        num_params_full = 0
        for l in self.full_layers:
            num_params_full += self.layers[l].num_params

        num_params_conv = 0
        for l in self.conv_layers:
            num_params_conv += self.layers[l].num_params

        return (num_params_full, num_params_conv)

    def sum_entropy(self, device):
        """
        Compute the joint entropy of the BQ-CNN.
        """
        sum_entropy_full = torch.tensor(0., 
            dtype = torch.float32, device = device)
        for l in self.full_layers:
            sum_entropy_full += self.layers[l].entropy()

        sum_entropy_conv = torch.tensor(0., 
            dtype = torch.float32, device = device)
        for l in self.conv_layers:
            sum_entropy_conv += self.layers[l].entropy()

        return (sum_entropy_full, sum_entropy_conv)

    def forward(self, inputs, targets, mode = "analytic", reduction = "sum"):
        """
        Computation of Bayesian Quantized Convolutional Neural Network (BQ-CNN)
        """

        # Layer 1: 5 x 5 Convolution layer + 2 x 2 Pooling 
        # + Batch normalization + Sign activation
        inputs = self.layers["conv1"](inputs, mode = mode)
        inputs = self.layers["norm1"](inputs, mode = mode)
        inputs = self.layers["sign1"](inputs, mode = mode)
        inputs = self.layers["pool1"](inputs, mode = mode)

        # Layer 2: 5 x 5 Convolution Layer + 2 x 2 Pooling 
        # + Batch normalization + Sign activation
        inputs = self.layers["conv2"](inputs, mode = mode)
        inputs = self.layers["norm2"](inputs, mode = mode)
        inputs = self.layers["sign2"](inputs, mode = mode)
        inputs = self.layers["pool2"](inputs, mode = mode)

        # Flatten the feature maps into a vector
        if isinstance(inputs, tuple):
            inputs_mean, inputs_var = inputs
            inputs_mean = inputs_mean.view(inputs_mean.size(0), -1)
            inputs_var  = inputs_var .view(inputs_var. size(0), -1)
            inputs = (inputs_mean, inputs_var)
        else:
            inputs = inputs.view(inputs.size(0), -1)

        # Layer 3: Fully connected layer + Batch normalization + Sign activation
        inputs = self.layers["full3"](inputs, mode = mode)
        inputs = self.layers["norm3"](inputs, mode = mode)
        inputs = self.layers["sign3"](inputs, mode = mode)

        # Layer 4: Fully connected layer + Softmax activation
        inputs = self.layers["full4"](inputs, mode = mode)
        prob, nll = self.layers["softmax"](inputs, 
            targets, mode = mode, return_nll = reduction)

        return (prob, nll.reshape([1]))


## Bayesian Quantized VGG-like Network (BQ-VGG)
class BayesVGGLike(nn.Module):
    def __init__(self, image_channels = 3, image_height = 32, image_width = 32,
        output_classes = 10, use_bias = True, quantization = "binary", pooling = "max"):
        """
        Initialization of Bayesian Quantized VGG-like Network (BQ-VGG).
        """
        super(BayesVGGLike, self).__init__()
        self.layers = nn.ModuleDict()

        # type of pooling layers
        assert pooling in ["avg", "max"], \
            "The type of pooling layer is not supported."

        BayesPool2d = {"avg": BayesAvgPool2d, 
                       "max": BayesMaxPool2d}[pooling]

        # Block 1: 2 x [3 x 3 Convolution layer + (2 x 2 Pooling layer)
        # + Batch normalization  + Sign activation]
        self.layers["conv1_1"] = BayesConv2d(image_channels, 128,
            kernel_size = 3, stride = 1, padding = 1, use_bias = use_bias,
            quantization = quantization, input_type = "dirac")
        self.layers["norm1_1"] = BayesBatchNorm2d(128)
        self.layers["sign1_1"] = BayesSign()

        self.layers["conv1_2"] = BayesConv2d(128, 128, 
            kernel_size = 3, stride = 1, padding = 1, use_bias = use_bias, 
            quantization = quantization, input_type = "discrete")
        self.layers["pool1"]   = BayesPool2d(kernel_size = 2)
        self.layers["norm1_2"] = BayesBatchNorm2d(128)
        self.layers["sign1_2"] = BayesSign()
        
        # Block 2: 2 x [3 x 3 Convolution layer + (2 x 2 Pooling layer) 
        # + Batch normalization  + Sign activation]
        self.layers["conv2_1"] = BayesConv2d(128, 256, 
            kernel_size = 3, stride = 1, padding = 1, use_bias = use_bias, 
            quantization = quantization, input_type = "discrete")
        self.layers["norm2_1"] = BayesBatchNorm2d(256)
        self.layers["sign2_1"] = BayesSign()

        self.layers["conv2_2"] = BayesConv2d(256, 256, 
            kernel_size = 3, stride = 1, padding = 1, use_bias = use_bias, 
            quantization = quantization, input_type = "discrete")
        self.layers["pool2"]   = BayesPool2d(kernel_size = 2)
        self.layers["norm2_2"] = BayesBatchNorm2d(256)
        self.layers["sign2_2"] = BayesSign()

        # Block 3: 2 x [3 x 3 Convolution layer + (2 x 2 Pooling layer) 
        # + Batch normalization  + Sign activation]
        self.layers["conv3_1"] = BayesConv2d(256, 512, 
            kernel_size = 3, stride = 1, padding = 1, use_bias = use_bias, 
            quantization = quantization, input_type = "discrete")
        self.layers["norm3_1"] = BayesBatchNorm2d(512)
        self.layers["sign3_1"] = BayesSign()

        self.layers["conv3_2"] = BayesConv2d(512, 512, 
            kernel_size = 3, stride = 1, padding = 1, use_bias = use_bias, 
            quantization = quantization, input_type = "discrete")
        self.layers["pool3"]   = BayesPool2d(kernel_size = 2)
        self.layers["norm3_2"] = BayesBatchNorm2d(512)
        self.layers["sign3_2"] = BayesSign()
        
        # Flatten feature maps into a vector
        input_length = (image_height // 8) * (image_width // 8) * 512

        # Block 4: (FC-layer + Sign activation) + (FC-layer + Softmax activation)
        self.layers["full4_1"] = BayesLinear(input_length, 1024, 
            use_bias = use_bias, quantization = quantization, input_type = "discrete")
        self.layers["norm4_1"] = BayesBatchNorm1d(1024)
        self.layers["sign4_1"] = BayesSign()

        self.layers["full4_2"] = BayesLinear(1024, output_classes, 
            use_bias = use_bias, quantization = quantization, input_type = "discrete")
        self.layers["softmax"] = BayesSoftmax(units = 1024, scale = 1)

        # names of the fully-connected and convolutional layers
        self.full_layers = ["full4_1", "full4_1"]
        self.conv_layers = ["conv1_1", "conv1_2", "conv2_1", 
                            "conv2_2", "conv3_1", "conv3_2"]

    def num_params(self):
        """
        Compute the total number of parameters in the BQ-VGG.
        """
        num_params_full = 0
        for l in self.full_layers:
            num_params_full += self.layers[l].num_params

        num_params_conv = 0
        for l in self.conv_layers:
            num_params_conv += self.layers[l].num_params

        return (num_params_full, num_params_conv)

    def sum_entropy(self, device):
        """
        Compute the joint entropy of the BQ-VGG.
        """
        sum_entropy_full = torch.tensor(0., 
            dtype = torch.float32, device = device)
        for l in self.full_layers:
            sum_entropy_full += self.layers[l].entropy()

        sum_entropy_conv = torch.tensor(0., 
            dtype = torch.float32, device = device)
        for l in self.conv_layers:
            sum_entropy_conv += self.layers[l].entropy()

        return (sum_entropy_full, sum_entropy_conv)

    def forward(self, inputs, targets, mode = "analytic", reduction = "sum"):
        """
        Computation of Bayesian Quantized VGG-like Network (BQ-VGG).
        """

        # Block 1: 2 x [3 x 3 Convolution layer + (2 x 2 Pooling layer) 
        # + Batch normalization  + Sign activation]
        inputs = self.layers["conv1_1"](inputs, mode = mode)
        inputs = self.layers["norm1_1"](inputs, mode = mode)
        inputs = self.layers["sign1_1"](inputs, mode = mode)

        inputs = self.layers["conv1_2"](inputs, mode = mode) 
        inputs = self.layers["pool1"]  (inputs, mode = mode)
        inputs = self.layers["norm1_2"](inputs, mode = mode)
        inputs = self.layers["sign1_2"](inputs, mode = mode)

        # Block 2: 2 x [3 x 3 Convolution layer + (2 x 2 Pooling layer) 
        # + Batch normalization  + Sign activation]
        inputs = self.layers["conv2_1"](inputs, mode = mode)
        inputs = self.layers["norm2_1"](inputs, mode = mode)
        inputs = self.layers["sign2_1"](inputs, mode = mode)

        inputs = self.layers["conv2_2"](inputs, mode = mode)
        inputs = self.layers["pool2"]  (inputs, mode = mode)
        inputs = self.layers["norm2_2"](inputs, mode = mode)
        inputs = self.layers["sign2_2"](inputs, mode = mode)

        # Block 3: 2 x [3 x 3 Convolution layer + (2 x 2 Pooling layer)
        # + Batch normalization  + Sign activation]
        inputs = self.layers["conv3_1"](inputs, mode = mode)
        inputs = self.layers["norm3_1"](inputs, mode = mode)
        inputs = self.layers["sign3_1"](inputs, mode = mode)

        inputs = self.layers["conv3_2"](inputs, mode = mode) 
        inputs = self.layers["pool3"]  (inputs, mode = mode)
        inputs = self.layers["norm3_2"](inputs, mode = mode)
        inputs = self.layers["sign3_2"](inputs, mode = mode)

        # Flatten feature maps into a vector
        inputs = inputs.view(inputs.size(0), -1)

        # Block 4: (FC-layer + Sign activation) + (FC-layer + Softmax activation)
        inputs = self.layers["full4_1"](inputs, mode = mode)
        inputs = self.layers["norm4_1"](inputs, mode = mode)
        inputs = self.layers["sign4_1"](inputs, mode = mode)

        inputs = self.layers["full4_2"](inputs, mode = mode)
        prob, nll = self.layers["softmax"](inputs, 
            targets, mode = mode, return_nll = reduction)

        return (prob, nll.reshape([1]))



## Bayesian Quantized VGG-like Network (BQ-VGG)
class BayesVGGLike_(nn.Module):
    def __init__(self, image_channels = 3, image_height = 32, image_width = 32,
        output_classes = 10, use_bias = True, quantization = "binary", pooling = "max"):
        """
        Initialization of Bayesian Quantized VGG-like Network (BQ-VGG).
        """
        super(BayesVGGLike_, self).__init__()
        self.layers = nn.ModuleDict()

        # type of pooling layers (and distribution of activation)
        BayesPool2d = {"avg_":  BayesAvgPool2d_, 
                       "prob_": BayesProbPool2d_}[pooling]

        activation_type = {"avg_":  'gaussian', 
                           "prob_": 'discrete'}[pooling]

        # Block 1: 2 x [3 x 3 Convolution layer + Batch normalization 
        # + Sign activation + (2 x 2 Pooling layer)]
        self.layers["conv1_1"] = BayesConv2d(image_channels, 128,
            kernel_size = 3, stride = 1, padding = 1, use_bias = use_bias,
            quantization = quantization, input_type = "dirac")
        self.layers["norm1_1"] = BayesBatchNorm2d(128)
        self.layers["sign1_1"] = BayesSign()

        self.layers["conv1_2"] = BayesConv2d(128, 128, 
            kernel_size = 3, stride = 1, padding = 1, use_bias = use_bias, 
            quantization = quantization, input_type = "discrete")
        self.layers["norm1_2"] = BayesBatchNorm2d(128)
        self.layers["sign1_2"] = BayesSign()

        self.layers["pool1"]   = BayesPool2d(kernel_size = 2)
        
        # Block 2: 2 x [3 x 3 Convolution layer + Batch normalization 
        # + Sign activation + (2 x 2 Pooling layer)]
        self.layers["conv2_1"] = BayesConv2d(128, 256, 
            kernel_size = 3, stride = 1, padding = 1, use_bias = use_bias, 
            quantization = quantization, input_type = activation_type)
        self.layers["norm2_1"] = BayesBatchNorm2d(256)
        self.layers["sign2_1"] = BayesSign()

        self.layers["conv2_2"] = BayesConv2d(256, 256, 
            kernel_size = 3, stride = 1, padding = 1, use_bias = use_bias, 
            quantization = quantization, input_type = "discrete")
        self.layers["norm2_2"] = BayesBatchNorm2d(256)
        self.layers["sign2_2"] = BayesSign()

        self.layers["pool2"]   = BayesPool2d(kernel_size = 2)

        # Block 3: 2 x [3 x 3 Convolution layer + Batch normalization 
        # + Sign activation + + (2 x 2 Pooling layer)]
        self.layers["conv3_1"] = BayesConv2d(256, 512,
            kernel_size = 3, use_bias = use_bias, stride = 1, padding = 1,
            quantization = quantization, input_type = activation_type)
        self.layers["norm3_1"] = BayesBatchNorm2d(512)
        self.layers["sign3_1"] = BayesSign()

        self.layers["conv3_2"] = BayesConv2d(512, 512,
            kernel_size = 3, stride = 1, padding = 1, use_bias = use_bias,
            quantization = quantization, input_type = "discrete")
        self.layers["pool3"]   = BayesPool2d(kernel_size = 2)
        self.layers["norm3_2"] = BayesBatchNorm2d(512)
        self.layers["sign3_2"] = BayesSign()
        
        # Flatten feature maps into a vector
        input_length = (image_height // 8) * (image_width // 8) * 512

        # Block 4: (FC-layer + Sign activation) + (FC-layer + Softmax activation)
        self.layers["full4_1"] = BayesLinear(input_length, 1024, 
            use_bias = use_bias, quantization = quantization, input_type = activation_type)
        self.layers["norm4_1"] = BayesBatchNorm1d(1024)
        self.layers["sign4_1"] = BayesSign()

        self.layers["full4_2"] = BayesLinear(1024, output_classes, 
            use_bias = use_bias, quantization = quantization, input_type = "discrete")
        self.layers["softmax"] = BayesSoftmax(units = 1024, scale = 1)

        # names of the fully-connected and convolutional layers
        self.full_layers = ["full4_1", "full4_1"]
        self.conv_layers = ["conv1_1", "conv1_2", "conv2_1", 
                            "conv2_2", "conv3_1", "conv3_2"]

    def num_params(self):
        """
        Compute the total number of parameters in the BQ-VGG.
        """
        num_params_full = 0
        for l in self.full_layers:
            num_params_full += self.layers[l].num_params

        num_params_conv = 0
        for l in self.conv_layers:
            num_params_conv += self.layers[l].num_params

        return (num_params_full, num_params_conv)

    def sum_entropy(self, device):
        """
        Compute the joint entropy of the BQ-VGG.
        """
        sum_entropy_full = torch.tensor(0., 
            dtype = torch.float32, device = device)
        for l in self.full_layers:
            sum_entropy_full += self.layers[l].entropy()

        sum_entropy_conv = torch.tensor(0., 
            dtype = torch.float32, device = device)
        for l in self.conv_layers:
            sum_entropy_conv += self.layers[l].entropy()

        return (sum_entropy_full, sum_entropy_conv)

    def forward(self, inputs, targets, mode = "analytic", reduction = "sum"):
        """
        Computation of Bayesian Quantized VGG-like Network (BQ-VGG).
        """

        # Block 1: 2 x [3 x 3 Convolution layer + Batch normalization 
        # + Sign activation + (2 x 2 Pooling layer)]
        inputs = self.layers["conv1_1"](inputs, mode = mode)
        inputs = self.layers["norm1_1"](inputs, mode = mode)
        inputs = self.layers["sign1_1"](inputs, mode = mode)

        inputs = self.layers["conv1_2"](inputs, mode = mode) 
        inputs = self.layers["norm1_2"](inputs, mode = mode)
        inputs = self.layers["sign1_2"](inputs, mode = mode)

        inputs = self.layers["pool1"]  (inputs, mode = mode)

        # Block 2: 2 x [3 x 3 Convolution layer + Batch normalization 
        # + Sign activation + (2 x 2 Pooling layer)]
        inputs = self.layers["conv2_1"](inputs, mode = mode)
        inputs = self.layers["norm2_1"](inputs, mode = mode)
        inputs = self.layers["sign2_1"](inputs, mode = mode)

        inputs = self.layers["conv2_2"](inputs, mode = mode)
        inputs = self.layers["norm2_2"](inputs, mode = mode)
        inputs = self.layers["sign2_2"](inputs, mode = mode)

        inputs = self.layers["pool2"]  (inputs, mode = mode)

        # Block 3: 2 x [3 x 3 Convolution layer + Batch normalization 
        # + Sign activation + + (2 x 2 Pooling layer)]
        inputs = self.layers["conv3_1"](inputs, mode = mode)
        inputs = self.layers["norm3_1"](inputs, mode = mode)
        inputs = self.layers["sign3_1"](inputs, mode = mode)

        inputs = self.layers["conv3_2"](inputs, mode = mode)
        inputs = self.layers["norm3_2"](inputs, mode = mode)
        inputs = self.layers["sign3_2"](inputs, mode = mode)

        inputs = self.layers["pool3"]  (inputs, mode = mode)

        # Flatten the feature maps into a vector
        if isinstance(inputs, tuple):
            inputs_mean, inputs_var = inputs
            inputs_mean = inputs_mean.view(inputs_mean.size(0), -1)
            inputs_var  = inputs_var .view(inputs_var. size(0), -1)
            inputs = (inputs_mean, inputs_var)
        else:
            inputs = inputs.view(inputs.size(0), -1)

        # Block 4: (FC-layer + Sign activation) + (FC-layer + Softmax activation)
        inputs = self.layers["full4_1"](inputs, mode = mode)
        inputs = self.layers["norm4_1"](inputs, mode = mode)
        inputs = self.layers["sign4_1"](inputs, mode = mode)

        inputs = self.layers["full4_2"](inputs, mode = mode)
        prob, nll = self.layers["softmax"](inputs, 
            targets, mode = mode, return_nll = reduction)

        return (prob, nll.reshape([1]))