import numpy as np


class Activation:
    def normal(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class SigmoidActivation(Activation):
    def normal(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1.0 - x)


class ReLuActivation(Activation):
    def normal(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        #derivative is 0 when x < 0 and x when x > 0; "undefined in x == 0"
        return x * (x > 0.0)

# class LeakyReLuActivation(Activation):
#     def normal(self, x):
#         return np.maximum(x * 0.01, x)

#     def derivative(self, x):
#         # uncertain derivative
#         for i in range (0, len(x)):
#             if i > 0.0:
#                 x[i] = x[i]
#             else:
#                 x[i] = 0.01
            
#         return x


class SoftmaxActivation(Activation):
    def normal(self, x):
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True) #idk
