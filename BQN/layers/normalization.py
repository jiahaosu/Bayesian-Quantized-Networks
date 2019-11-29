import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.utils as util

# TODO: check contiguous in THNN
# TODO: use separate backend functions?
class _BatchNorm(nn.Module):
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine']

    def __init__(self, num_features, eps = 1e-5, momentum = 0.1, affine = True,
                 track_running_stats = True):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias   = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum 
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_BatchNorm, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class _BayesBatchNorm(_BatchNorm):

    def forward(self, inputs, mode = "analytic"):
        """
        Computation of Bayesian BatchNorm layer.

        Arguments:
        ----------
        inputs: a pair of 4th-order tensors / a signle 4th-order tensor
            of size [batch_size, input_channels, input_height, input_width]
            "analytic": a pair of 4th-order tensors (inputs_mean, inputs_var)
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
            of size [batch_size, input_channels, input_height, input_width]
            "analytic": a pair of 4th-order tensors (outputs_mean, outputs_var) 
                Gaussian distribution of the output units.
                outputs_mean[b, j, x, y] = E[out[b, j, x, y]]
                outputs_var [b, j, x, y] = V[out[b, j, x, y]]
            "sampling" or "MAP": a 4th-order tensor 
                Realization of the output units.
        """
        if mode == "analytic":
            inputs_mean, inputs_var = inputs
            self._check_input_dim(inputs_mean)
            self._check_input_dim(inputs_var)

            inputs_mean = inputs_mean.transpose(0, 1)
            inputs_var  = inputs_var. transpose(0, 1)

            shape = inputs_mean.shape

            inputs_mean = inputs_mean.contiguous().view(shape[0], -1)
            inputs_var  = inputs_var. contiguous().view(shape[0], -1)

            mean = torch.mean( inputs_mean, dim = -1)
            var  = torch.mean((inputs_mean - mean.view(-1, 1)) ** 2 + inputs_var, dim = -1) 

            if self.training is not True: 
                outputs_mean = inputs_mean  -  self.running_mean.view(-1, 1)
                outputs_mean = outputs_mean / (self.running_var. view(-1, 1) ** 0.5 + self.eps)

                outputs_var  = inputs_var / (self.running_var.view(-1, 1) + self.eps)

            else:
                if self.track_running_stats is True:
                    with torch.no_grad():
                        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                        self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var

                outputs_mean = inputs_mean  - mean.view(-1, 1)
                outputs_mean = outputs_mean / (var.view(-1, 1) ** 0.5 + self.eps)

                outputs_var  = inputs_var / (var.view(-1, 1) + self.eps)

            outputs_mean =  self.weight.view(-1, 1) * outputs_mean + self.bias.view(-1, 1)
            outputs_var  = (self.weight.view(-1, 1) ** 2) * outputs_var

            outputs_mean = outputs_mean.view(shape).transpose(0, 1)
            outputs_var  = outputs_var. view(shape).transpose(0, 1)

            return (outputs_mean, outputs_var)

        else: # if mode in ["MAP", "sampling"]:
            self._check_input_dim(inputs)

            inputs = inputs.transpose(0, 1)
            shape  = inputs.shape

            inputs = inputs.contiguous().view(shape[0], -1)

            outputs = inputs  -  self.running_mean.view(-1, 1)
            outputs = outputs / (self.running_var. view(-1, 1) ** 0.5 + self.eps)

            outputs = self.weight.view(-1, 1) * outputs + self.bias.view(-1, 1)

            outputs = outputs.view(shape).transpose(0, 1)

            return outputs

class BayesBatchNorm1d(_BayesBatchNorm):

    def _check_input_dim(self, inputs):
        if inputs.dim() != 2 and inputs.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(inputs.dim()))

class BayesBatchNorm2d(_BayesBatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))



