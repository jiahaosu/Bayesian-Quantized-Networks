import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.utils as utils

import math

"""
This library consists of three types of 2D-pooling layers.
(1) Probabilistic Pooling (after sign activation)
(2) Maximum pooling (before / after sign activation) 
(3) Average Pooling (before / after sign activation)
""" 

# (1) Bayesian Probabilistic 2D-Pooling Layer
class BayesProbPool2d_(nn.Module):
    def __init__(self, kernel_size = 2):
        """
        Initialization of Bayesian Probabilistic 2D-pooling Layer.
        Note: the layer is applied after sign activation. 

        Arguments:
        ----------
        kernel_size: int or (int, int)
            The height/width of the pooling window.

        (TODO: more features are under development)
        """
        super(BayesProbPool2d_, self).__init__()

        self.kernel_height, self.kernel_width = utils._pair(kernel_size)

    def forward(self, inputs, mode = "analytic"):
        """
        Computation of Bayesian Probabilistic 2D-pooling Layer.

        Arguments:
        ----------
        inputs: a 4th-order tensor of size [batch_size, channels, input_height, input_width]
            1) Marginal distribution of the input feature maps.
            2) Realization of the input feature map.

        mode: str (options: "analytic", "sampling" or "MAP")
            Computation mode of the Bayesian pooling layer.
                "analytic": The inputs to the layer follow Bernoulli distribution.
                "sampling" or "MAP": The inputs to the layer are realizations.
            Note: The naming of the modes is consistent with linear layers.
            default: "analytic"

        Returns:
        --------
        outputs: a 4th-order tensor of size [batch_size, channels, output_height, output_width]
            1) Marginal distribution of the output feature maps.
            2) Realization of the output feature map.

        Note: output_height = (input_height + pad_height) / kernel_height
              output_width  = (input_width  + pad_width)  / kernel_width
        """

        # # TODO: add padding to the inputs
        # assert input_height % self.kernel_height == 0 and input_width % self.kernel_width == 0, \
        #   "The input height and/or width is not divisible by kernel_size."

        if mode == "analytic":
            outputs = F.avg_pool2d(inputs, (self.kernel_height, self.kernel_width))

        else: # if mode in ["MAP", "sampling"]:

            # 4th-order: batch_size(0) x channels(1) x input_height(2) x input_width(3)
            (batch_size, channels, input_height, input_width) = inputs.size()

            # compute height and width of the output feature maps
            output_height = input_height // self.kernel_height
            output_width  = input_width  // self.kernel_width

            kernel_area = self.kernel_height * self.kernel_width
            output_area = output_height * output_width

            # fold the inputs into [batch_size, channels * kernel_area, output_area]
            inputs = F.unfold(inputs, kernel_size = (self.kernel_height, self.kernel_width),
                                           stride = (self.kernel_height, self.kernel_width))

            # reshape the inputs into [batch_size, channels, kernel_area, output_area]
            inputs = inputs.view(batch_size, channels, kernel_area, output_area)

            # permute the inputs into [batch_size, channels, output_area, kernel_area]
            inputs = inputs.permute(0, 1, 3, 2).contiguous()

            # reshape the inputs into [batch_size * channels * output_area, kernel_area]
            num_patches = batch_size * channels * output_area
            inputs = inputs.view(num_patches, kernel_area)

            # sample uniformly from the inputs, returning a vector of length [batch_size * channels * output_area]
            outputs = inputs[range(num_patches), torch.randint(0, kernel_area, (num_patches,))]

            # reshape the outputs into [batch_size, channels, output_height, output_width]
            outputs = outputs.view(batch_size, channels, output_height, output_width)
            
        return outputs


# (2.2) Bayesian Maximum 2D-Pooling Layer
def phi(x):
    return torch.exp(-x**2/2) / math.sqrt(2 * math.pi)
        
def Phi(x):
    return (1 + torch.erf(x / math.sqrt(2))) / 2

class BayesMaxPool2d(nn.Module):
    def __init__(self, kernel_size = 2):
        """
        Initialization of Bayesian Maximum 2D-Pooling Layer.
        Note: the layer is applied before sign activation.

        Arguments:
        ----------
        kernel_size: int or (int, int)
            The height/width of the pooling window.

        (TODO: more features of pooling is under development)
        """
        super(BayesMaxPool2d, self).__init__()

        self.kernel_height, self.kernel_width = utils._pair(kernel_size)

    def forward(self, inputs, mode = "analytic"):
        """
        Computation of Bayesian Maximum 2D-Pooling Layer.

        Arguments:
        ----------
        inputs: a pair of 4th-order tensors of size [batch_size, channels, input_height, input_width]
            "analytic": Marginal distribution of the input feature maps.
            "sampling" or "MAP": Realizations of the input feature maps.

        mode: str (options: "analytic", "sampling" or "MAP")
            Computation mode of the Bayesian layer.
            "analytic": The inputs to the layer follow Bernoulli distribution.
            "sampling" or "MAP": The inputs to the layer are realizations.
            Note: The naming of the modes is consistent with linear layers.
            default: "analytic"

        Returns:
        --------
        outputs: a 4th-order tensor of size [batch_size, channels, output_height, output_width]
            1) Marginal distribution of the output feature maps.
            2) Realization of the output feature map.

        Note: output_height = (input_height + pad_height) / kernel_height
              output_width  = (input_width  + pad_width)  / kernel_width
        """

        # # TODO: add padding to the inputs
        # assert input_height % self.kernel_height == 0 and input_width % self.kernel_width == 0, \
        #   "The input height and/or width is not divisible by kernel_size."

        if mode == "analytic":
            inputs_mean, inputs_var = inputs
            shape = inputs_mean.shape

            # Step 1: pooling over the vertical axis
            inputs_mean = inputs_mean.view(shape[0], shape[1], shape[2], shape[3] // 2, 2)
            inputs_var  = inputs_var. view(shape[0], shape[1], shape[2], shape[3] // 2, 2)

            inputs_mean_0, inputs_mean_1 = inputs_mean[..., 0], inputs_mean[..., 1]
            inputs_var_0,  inputs_var_1  = inputs_var[ ..., 0], inputs_var[ ..., 1]

            alpha = torch.sqrt(inputs_var_0 + inputs_var_1)
            beta  = (inputs_mean_0 - inputs_mean_1) / alpha

            inters_mean = inputs_mean_0 * Phi(beta) + inputs_mean_1 * Phi(-beta) + alpha * phi(beta) 
            inters_var  = (inputs_mean_0 ** 2 + inputs_var_0) * Phi( beta) + \
                          (inputs_mean_1 ** 2 + inputs_var_1) * Phi(-beta) + \
                          (inputs_mean_0 + inputs_mean_1) * alpha * phi(beta) - inters_mean ** 2 

            # Step 2: pooling over the horizontal axis
            inters_mean = inters_mean.view(shape[0], shape[1], shape[2] // 2, 2, shape[3] // 2)
            inters_var  = inters_var. view(shape[0], shape[1], shape[2] // 2, 2, shape[3] // 2)

            inters_mean_0, inters_mean_1 = inters_mean[..., 0, :], inters_mean[..., 1, :]
            inters_var_0,  inters_var_1  = inters_var[ ..., 0, :], inters_var[ ..., 1, :]

            alpha = torch.sqrt(inters_var_0 + inters_var_1)
            beta  = (inters_mean_0 - inters_mean_1) / alpha

            outputs_mean = inters_mean_0 * Phi(beta) + inters_mean_1 * Phi(-beta) + alpha * phi(beta) 
            outputs_var  = (inters_mean_0 ** 2 + inters_var_0) * Phi( beta) + \
                           (inters_mean_1 ** 2 + inters_var_1) * Phi(-beta) + \
                           (inters_mean_0 + inters_mean_1) * alpha * phi(beta) - outputs_mean ** 2 

            outputs = (outputs_mean, outputs_var)

        else: # if mode in ["MAP", "sampling"]:
            outputs = F.max_pool2d(inputs, (self.kernel_height, self.kernel_width))

        return outputs


# (2.2) Bayesian Maximum 2D-Pooling Layer
class BayesMaxPool2d_(nn.Module):
    def __init__(self, kernel_size = 2):
        """
        Initialization of Bayesian Maximum 2D-Pooling Layer.
        Note: the layer is applied after sign activation.

        Arguments:
        ----------
        kernel_size: int or (int, int)
            The height/width of the pooling window.

        (TODO: more features of pooling is under development)
        """
        super(BayesMaxPool2d, self).__init__()

        self.kernel_height, self.kernel_width = utils._pair(kernel_size)

    def forward(self, inputs, mode = "analytic"):
        """
        Computation of Bayesian Maximum 2D-Pooling Layer.

        Arguments:
        ----------
        inputs: a 4th-order tensor of size [batch_size, channels, input_height, input_width]
            "analytic": Marginal distribution of the input feature maps.
            "sampling" or "MAP": Realizations of the input feature maps.

        mode: str (options: "analytic", "sampling" or "MAP")
            Computation mode of the Bayesian layer.
            "analytic": The inputs to the layer follow Bernoulli distribution.
            "sampling" or "MAP": The inputs to the layer are realizations.
            Note: The naming of the modes is consistent with linear layers.
            default: "analytic"

        Returns:
        --------
        outputs: a 4th-order tensor of size [batch_size, channels, output_height, output_width]
            1) Marginal distribution of the output feature maps.
            2) Realization of the output feature map.

        Note: output_height = (input_height + pad_height) / kernel_height
              output_width  = (input_width  + pad_width)  / kernel_width
        """

        # # TODO: add padding to the inputs
        # assert input_height % self.kernel_height == 0 and input_width % self.kernel_width == 0, \
        #   "The input height and/or width is not divisible by kernel_size."

        if mode == "analytic":
            # 4th-order: batch_size(0) x channels(1) x input_height(2) x input_width(3)
            batch_size, channels, input_height, input_width = inputs.size()

            # compute height/width of the output feature maps
            output_height = input_height // self.kernel_height
            output_width  = input_width  // self.kernel_width

            kernel_area = self.kernel_height * self.kernel_width
            output_area = output_height * output_width

            # fold the inputs into [batch_size, channels * kernel_area, output_area]
            inputs = F.unfold(inputs, kernel_size = (self.kernel_height, self.kernel_width),
                                      stride = (self.kernel_height, self.kernel_width))

            # fold the inputs into [batch_size, channels, kernel_area, output_area]
            inputs = inputs.view(batch_size, channels, kernel_area, output_area)

            # product in the probability space, returning a tensor of size [batch_size, channels, output_area] 
            inputs = torch.prod(inputs, dim = 2, keepdim = False)

            # reshape the inputs into [batch_size, channels, output_height, output_width]
            outputs = inputs.view(batch_size, channels, output_height, output_width)

        else: # if mode in ["MAP", "sampling"]:
            outputs = F.max_pool2d(inputs, (self.kernel_height, self.kernel_width))

        return outputs


## (3.1) Bayesian Average 2D-Pooling Layer
class BayesAvgPool2d(nn.Module):

    def __init__(self, kernel_size = 2):
        """
        Initialization of Bayesian Average 2D-Pooling Layer.
        Note: this layer is applied before sign activation.

        Arguments:
        ----------
        kernel_size: int or (int, int)
            The height/width of the pooling window.
        """
        super(BayesAvgPool2d, self).__init__()

        # handler of the 2D avg-pooling opertor
        self.avg_pool2d = lambda inputs: \
            F.avg_pool2d(inputs, kernel_size = kernel_size)

    def forward(self, inputs, mode = "analytic"):
        """
        Computation of Bayesian 2D avg-pooling layer.

        Arguments:
        ----------
        inputs: a pair of 4th-order tensors of size [batch_size, channels, input_height, input_width]
            "analytic": Marginal distribution of the input feature maps.
            "sampling" or "MAP': Realizations of the input feature maps.

        mode: str (options: "analytic", "sampling" or "MAP")
            Computation mode of the Bayesian layer.
                "analytic": The inputs to the layer follow Bernoulli/Gaussian distribution.
                "sampling" or "MAP": The inputs to the layer are realizations.
            Note: The naming of the modes is consistent with linear layers.

        Returns:
        --------
        outputs: a pair of 4th-order tensors of size [batch_size, channels, output_height, output_width]
            "analytic": Marginal distribution of the output feature maps.
            "sampling" or "MAP": Realizations of the output feature maps.

        Note: output_height = (input_height + pad_height) / kernel_height
              output_width  = (input_width  + pad_width)  / kernel_width
        """

        if mode == "analytic":
            inputs_mean, inputs_var = inputs

            outputs_mean = self.avg_pool2d(inputs_mean)
            outputs_var  = self.avg_pool2d(inputs_var)

            # returns a pair of 4-th order tensors
            outputs = (outputs_mean, outputs_var)

        else: # if mode in ["MAP", "sampling"]:

            # returns a single 4-th order tensor
            outputs = self.avg_pool2d(inputs)

        return outputs


## (3.1) Bayesian Average 2D-Pooling Layer
class BayesAvgPool2d_(nn.Module):

    def __init__(self, kernel_size = 2):
        """
        Initialization of Bayesian Average 2D-Pooling Layer.
        Note: this layer is applied after sign activation.

        Arguments:
        ----------
        kernel_size: int or (int, int)
            The height/width of the pooling window.
        """
        super(BayesAvgPool2d_, self).__init__()

        # handler of the 2D avg-pooling opertor
        self.avg_pool2d = lambda inputs: \
            F.avg_pool2d(inputs, kernel_size = kernel_size)

    def forward(self, inputs, mode = "analytic"):
        """
        Computation of Bayesian 2D avg-pooling layer.

        Arguments:
        ----------
        inputs: a 4th-order tensor of size of size [batch_size, channels, input_height, input_width]
            "analytic": Marginal distribution of the input feature maps.
            "sampling" or "MAP': Realizations of the input feature maps.

        mode: str (options: "analytic", "sampling" or "MAP")
            Computation mode of the Bayesian layer.
                "analytic": The inputs to the layer follow Bernoulli/Gaussian distribution.
                "sampling" or "MAP": The inputs to the layer are realizations.
            Note: The naming of the modes is consistent with linear layers.

        Returns:
        --------
        outputs: a pair of 4th-order tensors of size [batch_size, channels, output_height, output_width]
            "analytic": Marginal distribution of the output feature maps.
            "sampling" or "MAP": Realizations of the output feature maps.

        Note: output_height = (input_height + pad_height) / kernel_height
              output_width  = (input_width  + pad_width)  / kernel_width
        """

        if mode == "analytic":
            inputs_mean = 1 - 2 * inputs
            inputs_var  = 1 - inputs_mean**2

            outputs_mean = self.avg_pool2d(inputs_mean)
            outputs_var  = self.avg_pool2d(inputs_var)

            # returns a pair of 4-th order tensors
            outputs = (outputs_mean, outputs_var)

        else: # if mode in ["MAP", "sampling"]:

            # returns a single 4-th order tensor
            outputs = self.avg_pool2d(inputs)

        return outputs
