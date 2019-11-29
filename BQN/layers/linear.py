import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.utils as utils

import numpy as np

## Part 1: Utilities for the linear layers

def moments_(probs, levels):
    """
    Compute mean and variance of a probability tensor.
    Note: the probability is represented in the orginal space.

    Arguments:
    ----------
    probs: an (N+1)-th order tensor of size [dim[0], ..., dim[N-1], num_levels]
        Marginal distribution of the weights/bias.
        Prob[w[i_0, ..., i_{N-1}] = levels[l]] = probs[i_0, ..., i_{N-1}, l]
    
    levels: a vector of length [num_levels]
        Quantized values of the weights/bias.
        e.g. [-1, 1] (binary), [-1, 0, 1] (ternary) or [-3, -1, 1, 3] (quanternary).

    Returns: 
    --------
    mean: an N-th order tensor of size [dim[0], ..., dim[N-1]]
        Mean of the weights/bias.
        mean[i_0, ..., i_{N-1}] = E[w[i_0, ..., i_{N-1}]]

    var:  an N-th order tensor of size [dim[0], ..., dim[N-1]]
        Variance of the weights/bias.
         var[i_0, ..., i_{N-1}] = V[w[i_0, ..., i_{N-1}]]
    """
    
    # E[W] = sum_x w(x) p(x)
    mean = torch.matmul(probs, levels)

    # V[W] = sum_x w(x)^2 p(x) - E[W]^2
    var  = torch.matmul(probs, levels**2) - mean**2

    # returns a pair of N-th order tensors
    return (mean, var)


def moments_log(log_probs, 
        log_levels, bias_levels, 
        log_squared_levels, bias_squared_levels):
    """
    Compute mean and variance of a probability tensor.
    Note: the probability is represented in log-space. 

    Arguments:
    ----------
    log_probs: an (N+1)-th order tensor of size [dim[0], ..., dim[N-1], levels]
        Marginal distribution of the weights/bias, represented in log-space.
          Prob[w[i_0, ..., i_{N-1}] = levels[l]] 
        = exp(log_probs[i_0, ..., i_{N-1}, l]) / sum_k [exp(log_probs[i_0, ..., i_{N-1}, k])]
 
    log_levels: a vector of length [num_levels]
        Quantized values of the weights/bias, represented in log-space.
        log_levels[l] = log(levels[l] - bias_levels), 
        where bias_levels < min_k levels[k].

    bias_levels: float
        The bias to compute the log_levels.
        Typically, bias_levels = min_k levels[k] - 1.

    log_squared_levels: a vector of lenght [num_levels]
        Square of the quantized values of the weights/bias, represented in log-space.
        log_squared_levels = log(levels[l]^2 - bias_squared_levels),
        where bias_squared_levels < min_k log_squared_levels.

    bias_squared_levels: float
        The bias to compute the log_squared_levels.
        Typically, bias_squared_levels = min_k levels[k]^2 - 1. 

    Returns:
    --------
    mean: an N-th order tensor of size [dim[0], ..., dim[N-1]]
        Mean of the weights/bias.
        mean[i_0, ..., i_{N-1}] = E[w[i_0, ..., i_{N-1}]]

    var:  an N-th order tensor of size [dim[0], ..., dim[N-1]]
        Variance of the weights/bias.
         var[i_0, ..., i_{N-1}] = V[w[i_0, ..., i_{N-1}]]
    """

    # E[W] = \sum_x w(x) p(x)
    mean = torch.exp(torch.logsumexp(log_probs + log_levels, dim = -1) - 
        torch.logsumexp(log_probs, dim = -1)) + bias_levels

    # E[W^2] = \sum_x w(x)^2 p(x)
    second_moment = torch.exp(torch.logsumexp(log_probs + log_squared_levels, dim = -1) - 
        torch.logsumexp(log_probs, dim = -1)) + bias_squared_levels

    # V[W] = E[W^2] - E[W]^2
    var  = second_moment - mean ** 2

    # returns a pair of N-th order tensors
    return (mean, var)


def entropy_(probs, reduction = "sum"):
    """
    Compute the entropy of a probability tensor.
    Note: the probability is represented in original space.

    Arguments:
    ----------
    probs: an (N+1)-th order tensor of size [dim[0], ..., dim[N-1], num_levels]
        Marginal distribution of the weights/bias.
        Prob[w[i_0, ..., i_{N-1}] = levels[l]] = probs[i_0, ..., i_{N-1}, l]

    reduction: str (options: "mean", "sum" or "none")
        Reduction mode on the output.
        default: "sum"

    Returns:
    --------
    entropy: a scalar / an N-th order tensor (depending on the reduction mode)
        Entropy of the probability tensor.
            "none": an N-th order tensor of [dim[0], ..., dim[N-1]].
            "sum"/"mean": a float scalar.
        Note: H(p) = - sum_w p(w) log(p(w)) (in probability space)
                   = - sum_w softmax(q(w)) log_softmax(q(w)) (in log-probability space)
    """

    # compute the elementwise entropy 
    outputs = - torch.log(probs) * probs 

    # reduce the outputs according to the reduction mode
    if reduction == "sum":
        outputs = torch.sum(outputs)
    elif reduction == "mean":
        outputs = torch.mean(outputs)
    elif reduction == "none":
        outputs = torch.sum(outputs, dim = -1)
    else:
        raise NotImplementedError

    return outputs


def entropy_log(log_probs, reduction = "sum"):
    """
    Compute the entropy of a probability tensor.
    Note: the probability is represented in log-space

    Arguments:
    ----------
    probs: an (N+1)-th order tensor of size [dim[0], ..., dim[N-1], num_levels]
        Marginal distribution of the weights/bias, represented in log-space.
          Prob[w[i_0, ..., i_{N-1}] = levels[l]] 
        = exp(log_probs[i_0, ..., i_{N-1}, l]) / sum_k [exp(log_probs[i_0, ..., i_{N-1}, k])]

    reduction: str ("mean", "sum" or "none")
        Reduction mode on the output.
        default: "sum"

    Returns:
    --------
    entropy: a scalar or an N-th order tensor (depending on the reduction mode)
        Entropy of the probability tensor.
            "none": an N-th order tensor of [dim[0], ..., dim[N-1]].
            "sum"/"mean": a float scalar.
        Note: H(p) = - sum_w p(w) log(p(w)) (in probability space)
                   = - sum_w softmax(q(w)) log_softmax(q(w)) (in log-probability space)
    """

    # compute the elementwise entropy 
    outputs = - F.log_softmax(log_probs, 
        dim = -1) * F.softmax(log_probs, dim = -1) 

    # reduce the outputs according to the reduction mode
    if reduction == "sum":
        outputs = torch.sum(outputs)
    elif reduction == "mean":
        outputs = torch.mean(outputs)
    elif reduction == "none":
        outputs = torch.sum(outputs, dim = -1)
    else:
        raise NotImplementedError

    return outputs


def sample(probs, levels, log_space = True):
    """
    Sampling from a probability tensor.

    Arguments:
    ----------
    probs: an (N+1)-th order tensor of size [dim[0], ..., dim[N-1], num_levels]
        Marginal distribution of the weights/bias.
    
    levels: a vector of length [num_levels]
        Quantized values of the weights/bias.
        Options: [-1, 1] (binary), [-1, 0, 1] (ternary) or [-3, -1, 1, 3] (quanternary)

    log_space: bool
        Whether the probability tensor is represented in log-space.
        If log_space is True:
            Prob[w[i_0, ..., i_{N-1}] = levels[l]] = probs[i_0, ..., i_{N-1}, l]
        If log_space is False:
            Prob[w[i_0, ..., i_{N-1}] = levels[l]] 
          = exp(log_probs[i_0, ..., i_{N-1}, l]) / sum_k [exp(log_probs[i_0, ..., i_{N-1}, k])]

    Returns:
    --------
    samples: an N-th order tensor of size [dim[0], ..., dim[N-1]]
        Realization of the weights/bias.
        Pr[samples[i_0, ..., i_{N-1}] = levels[l]] = probs[i_0, ..., i_{N-1}, l]
    """

    # compute the probability from its log-space (if needed)
    if log_space: probs = F.softmax(probs, dim = -1)

    # obtain the shape of 
    shape, num_levels = list(probs.size()[:-1]), probs.size(-1)

    # reshape the probability tensor into [num_entries, num_levels] (for subsequent sampling)
    # Note: num_entries = batch_size * dim[0] * ... * dim[N-1] 
    probs = probs.view(-1, num_levels)

    # sampling from the weights according to the probability tensor
    indices = torch.squeeze(torch.multinomial(probs, num_samples = 1), dim = -1)
    samples = levels[indices]

    # reshape the samples tensor into [batch_size, dim[0], ..., dim[N-1]]
    samples = samples.view(shape)

    return samples 


## Part 2: Bayesian linear layers

# Bayesian fully-conencted layer
class BayesLinear(nn.Module):
    def __init__(self, input_units, output_units, quantization = "binary", 
        input_type = "discrete", use_bias = True):
        """
        Initialization of Bayesian fully-connected layer.

        Arguments:
        ----------
        input_units: int
            The number of input units.
        output_units: int
            The number of output units.

        quantization: str / list of floats of length [num_levels]
            Quantiation levels of the weights/bias.
                "binary" :     the weights/bias are quantized into [-1, 1].
                "ternary":     the weights/bias are quantized into [-1, 0, 1].
                "quanternary": the weigtht/bias are quantized into [-3,-1, 1, 3].
            default: "binary"
            Alternatively, the levels can be set manually with a list of floats.

        input_type: str (options: "dirac", "discrete" or "gaussian")
            The distribution type of the inputs to the fully-connected layer.
                "dirac": the inputs follow dirac distribution (as model input).
                "discrete": the inputs follow Bernoulli distribution (after activation).
                "gaussian": the inputs follow Gaussian distribution (before activation).
            default: "discrete"

        use_bias: bool
            Whether or not to add the bias to the output.
            default: True
        """
        super(BayesLinear, self).__init__()

        ## (1) input and output interfaces

        # distribution type of input data
        self.input_type = input_type
        assert input_type in ["dirac", "discrete", "gaussian"], \
            "The distribution type of the inputs is not supported."

        # whether to add bias to the output
        self.use_bias = use_bias

        # number of parameters in the layer
        self.num_params = input_units * output_units
        if self.use_bias: 
            self.num_params += output_units

        ## (2) Quantization levels
        if isinstance(quantization, (list, tuple, np.ndarray)):
            levels = quantization
        elif isinstance(quantization, str):
            levels = {"binary": [-1, 1], "ternary": [-1, 0, 1], 
                "quanternary": [-3, -1, 1, 3]}[quantization]
        else:
            raise NotImplementedError

        levels = np.float32(levels)
        num_levels = len(levels)

        # quantized values of the weights/bias, in original space
        self.levels = nn.Parameter(torch.Tensor(levels), requires_grad = False)

        # quantized values in log-space
        bias_levels = np.min(levels) - 1.
        log_levels  = np.log(levels  - bias_levels)

        self.bias_levels = nn.Parameter(torch.Tensor([bias_levels]), requires_grad = False)
        self.log_levels  = nn.Parameter(torch.Tensor(log_levels),    requires_grad = False)

        # square of the quantized values in log-space
        bias_squared_levels = np.min(np.square(levels)) - 1.
        log_squared_levels  = np.log(np.square(levels) - bias_squared_levels)

        self.bias_squared_levels = nn.Parameter(torch.Tensor([bias_squared_levels]), requires_grad = False)
        self.log_squared_levels  = nn.Parameter(torch.Tensor(log_squared_levels),    requires_grad = False)
        
        ## (3) Initialize the learnable parameters
        self.log_probs = nn.Parameter(torch.Tensor(output_units, input_units, num_levels))
        torch.nn.init.xavier_uniform_(self.log_probs)

        if use_bias:
            self.log_probs_bias = nn.Parameter(torch.Tensor(output_units, num_levels))
            torch.nn.init.zeros_(self.log_probs_bias)
        else: 
            self.register_parameter("log_probs_bias", None)

    def entropy(self):
        """
        The joint entropy of the weights/bias in the layer.

        Returns:
        --------
        entropy: float
            The joint entropy of the weights/bias in the layer.
        """

        sum_entropy = entropy_log(self.log_probs)
        if self.use_bias:
            sum_entropy += entropy_log(self.log_probs_bias)

        return sum_entropy

    def forward(self, inputs, mode = "analytic"):
        """
        Computation of the Bayesian fully-connected layer.

        Arguments:
        ----------
        inputs: a pair of 2nd-order tensors / a single 2nd-order tensor
        If mode is "analytic": 
            "dirac": a 2nd-order tensor (dirac) of size [batch_size, input_units]
                Dirac distribution of the input units (as model inputs at the first layer).
                Prob[in[b, i] = dirac[b, i]] = infinity
            "discrete": a 2nd-order tensor (probs) of size [batch_size, input_units] 
                Discrete distribution of the input units (following sign activation).
                Prob[in[b, i] = -1] = probs[b, i] 
            "gaussian": a pair of 2nd-order tensors (inputs_mean, inputs_var) of size [batch_size, input_units]
                Gaussian distribution of the input units (following linear layers).
                inputs_mean[b, i] = E[in[b, i]]
                inputs_var [b, i] = V[in[b, i]]
        If mode is "sampling" or "MAP":
            a single 2nd-order tensor of size [batch_size, input_units]
                Realization of the input units.

        mode: str (options: "analytic", "sampling" or "MAP")
            The computational mode of the fully-connected layer.
            default: "analytic"

        Returns:
        --------
        outputs: a pair of 2nd-order tensors / a signle 2nd-order tensor
            "analytic": a pair of 2nd-order tensors (outputs_mean, outputs_var) of size [batch_size, output_units]
                Gaussian distribution of the output units.
                outputs_mean[b, j] = E[out[b, j]]
                outputs_var [b, j] = V[out[b, j]]
            "sampling" or "MAP": a single 2nd-order tensor of size [batch_size, output_units]
                Realization of the output units.
        """

        # Analytic mode: (approximated) marginalization of the weights/bias distribution
        if mode == "analytic":
            # mean and variance of the inputs
            if self.input_type == "dirac":
                inputs_mean = inputs
                inputs_var  = torch.ones_like(inputs) * 1e-2
            elif self.input_type == "discrete":
                inputs_mean = 1 - 2 * inputs
                inputs_var  = 1 - inputs_mean**2
            elif self.input_type == "gaussian":
                inputs_mean, inputs_var = inputs
            else: 
                raise NotImplementedError

            # mean and variance of the weights (and bias)
            (weights_mean, weights_var) = moments_log(
                self.log_probs, self.log_levels, self.bias_levels, 
                self.log_squared_levels, self.bias_squared_levels)

            (bias_mean, bias_var) = moments_log(
                self.log_probs_bias, self.log_levels, self.bias_levels, 
                self.log_squared_levels, self.bias_squared_levels) \
                    if self.use_bias else (None, None)

            # mean and variance of the outputs
            # Note: Y = WX, E[Y] = E[W] E[X], V[Y] = E[W]^2 V[X] + V[W] E[X]^2
            outputs_mean = F.linear(inputs_mean, weights_mean, bias_mean)
            outputs_var  = F.linear(inputs_var, weights_mean**2, bias_var) + \
                F.linear(inputs_mean**2 + inputs_var, weights_var)
            
            # returns a pair of 2nd-order tensors
            outputs = (outputs_mean, outputs_var)

        else: # if mode == "sampling" or mode == "MAP":

            # Sampling mode: sampling from the weights/bias distribution
            if mode == "sampling":
                weights_ = sample(self.log_probs, self.levels)
                bias_    = sample(self.log_probs_bias, self.levels) \
                    if self.use_bias else None

            # MAP mode: maximum a posterior of the weights/bias distribution
            elif mode == "MAP":
                weights_ = self.levels[torch.argmax(self.log_probs, dim = -1)]
                bias_    = self.levels[torch.argmax(self.log_probs_bias, dim = -1)] \
                    if self.use_bias else None
            else: 
                raise NotImplementedError

            # returns a single 2nd-order tensor
            outputs = F.linear(inputs, weights_, bias_)

        return outputs


class BayesConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, 
        quantization = "ternary", input_type = "discrete", use_bias = True,
        stride = 1, padding = 0, dilation = 1, groups = 1):
        """
        Initialization of Bayesian 2D-convolutional layer.

        Arguments:
        ----------
        input_channels: int
            Number of input channels.
        output_channels: int
            Number of output channels.
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.

        quantization: str / list of floats of length [num_levels]
            Quantiation levels of the weights/bias.
                "binary" :     the weights/bias are quantized into [-1, 1].
                "ternary":     the weights/bias are quantized into [-1, 0, 1].
                "quanternary": the weigtht/bias are quantized into [-3,-1, 1, 3].
            default: "ternary"
            Alternatively, the levels can be set manually with a list of floats.

        input_type: str (options: "dirac", "discrete" or "gaussian")
            The distribution type of the inputs to the fully-connected layer.
                "dirac": the inputs follow dirac distribution (as model input).
                "discrete": the inputs follow Bernoulli distribution (after activation).
                "gaussian": the inputs follow Gaussian distribution (before activation).
            default: "discrete"

        use_bias: bool
            Whether or not to add the bias to the output.
            default: True

        stride, padding, dilation, groups: 
            These hyper-parameters are inherited from Conv2d layer.
        """
        super(BayesConv2d, self).__init__()

        ## (1) input and output interfaces

        # distribution type of input data
        self.input_type = input_type
        assert input_type in ["dirac", "discrete", "gaussian"], \
            "The distribution type of the inputs to the layer is not supported."

        # whether to add bias to the output
        self.use_bias = use_bias

        # number of parameters in the layer
        input_channels = int(input_channels / groups)
        self.num_params = input_channels * output_channels * (kernel_size ** 2)
        if self.use_bias:
            self.num_params += output_channels

        ## (2) Handler of 2D convolutional operator
        self.conv2d = lambda inputs, weights, bias = None: F.conv2d(
            inputs, weights, bias = bias, stride = stride, 
            padding = padding, dilation = dilation, groups = groups)

        ## (3) Quantization levels
        if isinstance(quantization, (list, tuple, np.ndarray)):
            levels = quantization
        elif isinstance(quantization, str):
            levels = {"binary": [-1, 1], "ternary": [-1, 0, 1], 
                "quanternary": [-3, -1, 1, 3]}[quantization]
        else:
            raise NotImplementedError

        levels = np.float32(levels)
        num_levels = len(levels)

        # quantized values of the weights/bias, in original space
        self.levels = nn.Parameter(torch.Tensor(levels), requires_grad = False)

        # quantized values in log-space
        bias_levels = np.min(levels) - 1.
        log_levels  = np.log(levels  - bias_levels)

        self.bias_levels = nn.Parameter(torch.Tensor([bias_levels]), requires_grad = False)
        self.log_levels  = nn.Parameter(torch.Tensor(log_levels),    requires_grad = False)

        # square of the quantized values in log-space
        bias_squared_levels = np.min(np.square(levels)) - 1
        log_squared_levels  = np.log(np.square(levels)  - bias_squared_levels)

        self.bias_squared_levels = nn.Parameter(torch.Tensor([bias_squared_levels]), requires_grad = False)
        self.log_squared_levels  = nn.Parameter(torch.Tensor(log_squared_levels),    requires_grad = False)

        ## (4) Initialize the learnable parameters
        self.log_probs = nn.Parameter(torch.Tensor(output_channels, 
            input_channels, kernel_size, kernel_size, num_levels))
        torch.nn.init.xavier_uniform_(self.log_probs)

        if use_bias:
            self.log_probs_bias = nn.Parameter(torch.Tensor(output_channels, num_levels))
            torch.nn.init.zeros_(self.log_probs_bias)
        else: 
            self.register_parameter("params_bias", None)

    def entropy(self):
        """
        The joint entropy of the weights/bias in the Bayesian fully-connected layer.

        Returns:
        --------
        entropy: float
            The joint entropy of the weights/bias in the layer.
        """

        sum_entropy = entropy_log(self.log_probs)
        if self.use_bias:
            sum_entropy += entropy_log(self.log_probs_bias)

        return sum_entropy

    def forward(self, inputs, mode = "analytic"):
        """
        Computation of Bayesian 2D-convolutional layer.

        Arguments:
        -----------
        inputs: a pair of 4th-order tensors / a signle 4th-order tensor
        If mode is "analytic":
            "dirac": a 4th-order tensor (dirac) of size [batch_size, input_channels, input_height, input_width]
                Dirac distribution of the input features (as model inputs at the first layer).
                Prob[in[b, i] = dirac[b, i]] = infinity
            "discrete": a 4th-order tensor (probs) of size [batch_size, input_channels, input_height, input_width] 
                Discrete distribution of the input features (following sign activation).
                probs[b, i, x, y] = Prob[in[b, i, x, y] = -1]
            "gaussian": a pair of 4th-order tensors (inputs_mean, inputs_var) of size [batch_size, input_channels, height, width]
                Gaussian distribution of the input features (following linear layers).
                inputs_mean[b, i, x, y] = E[in[b, i, x, y]]
                inputs_var [b, i, x, y] = V[in[b, i, x, y]]
        If mode is "sampling" or "MAP":
            a 4th-order tensor of size [batch_size, input_channels, input_height, input_width]
                Realization of the input units.

        mode: str (options: "analytic", "sampling" or "MAP")
            The computational mode of the fully-connected layer.
            default: "analytic"
        
        Returns:
        --------
        outputs: a pair of 4th-order tensors / a signle 4th-order tensor
            "analytic": a pair of 4th-order tensors (outputs_mean, outputs_var) of size [batch_size, output_channels, output_height, output_width]
                Gaussian distribution of the output units.
                outputs_mean[b, j, x, y] = E[out[b, j, x, y]]
                outputs_var [b, j, x, y] = V[out[b, j, x, y]]
            "sampling" or "MAP": a 4th-order tensor of size [batch_size, output_channels, output_height, output_width]
                Realization of the output units.
        """

        # Analytic mode: (approximated) marginalization of the weights/bias distribution
        if mode == "analytic":
            # mean and variance of the inputs
            if self.input_type == "dirac":
                inputs_mean = inputs
                inputs_var  = torch.ones_like(inputs) * 1e-2
            elif self.input_type == "discrete":
                inputs_mean = 1 - 2 * inputs
                inputs_var  = 1 - inputs_mean**2
            else: # if self.input_type == "gaussian":
                inputs_mean, inputs_var = inputs

            # mean and variance of the weights (and bias)
            (weights_mean, weights_var) = moments_log(
                self.log_probs, self.log_levels, self.bias_levels, 
                self.log_squared_levels, self.bias_squared_levels)

            (bias_mean, bias_var) = moments_log(
                self.log_probs_bias, self.log_levels, self.bias_levels, 
                self.log_squared_levels, self.bias_squared_levels) \
                    if self.use_bias else (None, None)

            # mean and variance of the outputs
            # Note: Y = WX, E[Y] = E[W] E[X], V[Y] = E[W]^2 V[X] + V[W] E[X]^2
            outputs_mean = self.conv2d(inputs_mean, weights_mean, bias_mean)
            outputs_var  = self.conv2d(inputs_var, weights_mean**2, bias_var) + \
                self.conv2d(inputs_mean**2 + inputs_var, weights_var)

            # returns a pair of 4th-order tensors
            outputs = (outputs_mean, outputs_var)

        else: # if mode == "sampling" or mode == "MAP":

            # Sampling mode: sampling from the weights/bias distribution
            if mode == "sampling":
                weights_ = sample(self.log_probs, self.levels)
                bias_    = sample(self.log_probs_bias, self.levels) \
                    if self.use_bias else None

            # MAP mode: maximum a posterior of the weights/bias distribution
            elif mode == "MAP":
                weights_ = self.levels[torch.argmax(self.log_probs, dim = -1)]
                bias_    = self.levels[torch.argmax(self.log_probs_bias, dim = -1)] \
                    if self.use_bias else None
            else:
                raise NotImplementedError
            
            # returns a single 4th-ordre tensor
            outputs = self.conv2d(inputs, weights_, bias_)

        return outputs