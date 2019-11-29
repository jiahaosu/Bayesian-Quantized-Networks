import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.utils as utils

import math

# Bayesian Sign Layer
class BayesSign(nn.Module):

    def forward(self, inputs, mode = "analytic", eps = 1e-6):
        """
        Computation of Bayesian sign layer.

        Arguments:
        ----------
        inputs: a pair of (N+1)-th order tensors / a single (N+1)-th order tensor, for example
                2nd-order tensor(s) of size [batch_size, units] for fully-connected layer.
                4th-order tensor(s) of size [batch_size, channels, height, width] for 2D-convolutional layer.
            1) "analytic": Mean and variance of input units / feature maps.
                inputs_mean[b, i] = E[in[b, i]] / inputs_mean[b, i, x, y] = E[in[b, i, x, y]]
                inputs_var [b, i] = V[in[b, i]] / inputs_mean[b, i, x, y] = E[in[b, i, x, y]]
            2) "sampling" or "MAP": Realizations of the input units / feature maps.
                
        mode: str (options: "analytic", "sampling" or "MAP")
            Computation mode of the Bayesian sign layer.
                "analytic": The inputs to the layer are Gaussian distribution.
                "sampling" or "MAP": The inputs to the layer are realizations.
            Note: The naming of the modes is consistent with linear layers. 

        eps: float
            A small number to prevent dividing zero, e.g.
            a / b will be computed as a / (b + eps)
            default: 1e-6

        Returns:
        --------
        outputs: an (N+1)-th order tensor, with the same size of the inputs
            1) "analytic": Bernoulli distribution (probs) of output units/feature maps.
                Note: Pr[out[b, i_0, ..., i_{N-1}] = -1] = probs[b, i_0, i_1, ..., i_{N-1}] 
            2) "sampling" or "MAP": Realizations of the output units/feature maps.
        """

        if mode == "analytic":
            inputs_mean, inputs_var = inputs
            outputs = 0.5 * (1 - torch.erf(inputs_mean / (torch.sqrt(2 * inputs_var) + eps)))
        else: # if mode == "sampling" or mode == "MAP":
            outputs = torch.sign(inputs) 

        return outputs


# Bayesian Softmax Layer (for classification problem)
class BayesSoftmax(nn.Module):
    def __init__(self, units, scale = 1):
        """
        Initialization of Bayesian softmax layer.

        Arguments:
        ----------
        units: int
            The number of input units in PREVIOUS linear layer.

        scale: float
            Scaling factor of the logits. 
            Note: the logits are divided by scale s before the softmax function, i.e.
            prob_c = exp(logit_c / s) / sum_k (logit_k / s), where s = scale * previous_units
            default: 1
        """
        super(BayesSoftmax, self).__init__()

        # notice that scale is a learnable parameter, storing in log-space (to ensure it is positive)
        self.params_scale = nn.Parameter(torch.log(torch.Tensor([units * scale])))

    def forward(self, logits, targets, mode = "analytic", return_nll = "sum"):
        """
        Computation of Bayesian softmax layer.

        Arguments:
        ----------
        logits: a pair of 2nd-order tensor / a single 2nd-order tensor of size [num_classes, levels] 
            1) "analytic": Mean and variance of the input units. 
            2) "sampling" or "MAP": Realizations of the input units.

        targets: a vector of length [batch_size].
            where targets[i] is the label for inputs[i].

        mode: "analytic", "sampling" or "MAP"
            Computation mode of the Bayesian softmax layer.
                "analytic": The inputs to the layer follow Gaussian distribution.
                "sampling" or "MAP": The inputs to the layer are realizations.
            Note: The naming of the modes is consistent with linear layers. 
        
        return_nll: None, "sum", "mean", "none"
            Whether or not to return the negative log-likelihood, 
            (and if so) the reduction applied to the outputs
            default: "sum"

        Returns:
        --------
        prob: a 2nd-order tensor of size [batch_size, num_classes]
            Predictive distributions of the inputs.
            Therefore, sum_c prob[b, c] = 1

        nll: float
            Negative log-likelihood of the predictive distribution.
            Note: the return format depends on return_nll
        """

        # map the scale from log-space to normal space
        scale = torch.exp(self.params_scale)

        if mode == "analytic":
            logits_mean, logits_var = logits

            # Compute the probabilities over the labels
            # first-order approximation
            prob = F.softmax(logits_mean / scale, dim = -1)
            # second-order adjustment
            prob = prob + 1/(2 * scale*2) * (prob - 2 * prob**2) * \
                (logits_var - torch.sum(prob * logits_var, dim = 1, keepdim = True))

            # Compute the negative log-likelihood
            if return_nll is not None:
                logits_mean, logits_var = logits
                batch_size = logits_mean.size()[0]

                # zero-order approximation
                nll = -logits_mean[range(batch_size), targets] / scale
                # second-order adjustment 
                nll += torch.logsumexp(logits_mean / scale + logits_var / (scale**2), dim = 1)
                # reduce the nll to scalar according to reduction mode  
                if return_nll != "none":
                    nll = torch.mean(nll) if return_nll == "mean" else torch.sum(nll)

        else: # if mode == "sampling" or mode == "MAP":
            # Compute the probabilities over labels
            prob = F.softmax(logits / scale, dim = -1)

            # Compute the negative log-likelihood
            if return_nll is not None:
                nll = F.cross_entropy(logits / scale, targets, reduction = return_nll)

        return (prob, nll) if return_nll else (prob, None)


# Bayesian Gaussian Layer (for regression problem)
class BayesGaussian(nn.Module):
    def __init__(self, input_units, output_units = 1, scale = 1):
        """
        Initialization of Bayesian Gaussian layer.

        Arguments:
        ----------
        input_units: int
            The number of input units to the Gaussian layer.
        output_units: int
            The number of output units of the Gaussian layer.

        scale: float
            The intrinsic variance of predictive distribution.
            default: 1
        """
        super(BayesGaussian, self).__init__()

        # notice that 
        self.params = nn.Parameter(torch.Tensor(input_units, output_units)) 
        torch.nn.init.xavier_uniform_(self.params)

        # notice that scale is a learnable parameter, storing in log-space (to ensure it is positive)
        self.params_var = nn.Parameter(torch.log(torch.ones(output_units) * (input_units * scale)))

    def forward(self, inputs, targets, mode = "analytic", return_nll = "sum"):
        """
        Computation of Bayesian Gaussian layer.

        Arguments:
        ----------
        inputs: a pair of 2nd-order tensor / a single 2nd-order tensor of size [num_classes, levels] 
            1) "analytic": Mean and variance of the input units. 
            2) "sampling" or "MAP": Realizations of the input units.

        targets: a vector of length [batch_size].
            targets[i] is the target for inputs[i, :].

        mode: "analytic", "sampling" or "MAP"
            Computation mode of the Bayesian Gaussian layer.
                "analytic": The inputs to the layer follow Gaussian distribution.
                "sampling" or "MAP": The inputs to the layer are realizations.
            Note: The naming of the modes is consistent with linear layers. 
        
        return_nll: None, "sum", "mean", "none"
            Whether or not to return the loss for Bayesian learning,
            (and if so) the reduction applied to the outputs
            default: "sum"

        Returns:
        --------
        prob: a pair of 2nd-order tensor of size [batch_size]
            Predictive distributions in terms of mean and variance.

        nll: float
            The loss function for Bayesian learning. 
            Note: the return format depends on return_nll.
        """

        # map the variance from log-space to normal space
        var = torch.exp(self.params_var)

        if mode == "analytic":
            inputs_mean = 1 - 2 * inputs
            inputs_var  = 1 - inputs_mean**2

            # Compute the mean and variance over the predictions
            outputs_mean = torch.matmul(inputs_mean, self.params)
            outputs_var_ = torch.matmul(inputs_var,  self.params ** 2)
            outputs_var  = outputs_var_ + var

            outputs = (outputs_mean, outputs_var)

            # Compute the negative log-likelihood
            if return_nll is not None:
                nll = ((targets - outputs_mean) ** 2 + outputs_var_) / (2 * var) \
                    + torch.log(2 * math.pi * var) / 2

                if return_nll != "none":
                    nll = torch.mean(nll) if return_nll == "mean" else torch.sum(nll)

        else: # if mode == "sampling" or mode == "MAP":
            outputs = torch.matmul(inputs, self.params)
            outputs = outputs + torch.sqrt(var) * torch.randn_like(outputs)

            # Compute the negative log-likelood
            if return_nll is not None:
                nll = ((targets - outputs) ** 2) / (2 * var) \
                    + torch.log(2 * math.pi * var) / 2 

                if return_nll != "none":
                    nll = torch.mean(nll) if return_nll == "mean" else torch.sum(nll)

        return (outputs, nll) if return_nll else (outputs, None)