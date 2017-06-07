from rllab.misc import ext
from rllab.misc import krylov
from rllab.misc import logger
from rllab.core.serializable import Serializable
import theano.tensor as TT
import theano
import itertools
import numpy as np
from rllab.misc.ext import sliced_fun
from _ast import Num
import scipy




class nonlinearConjugateGradientOptimizer(Serializable):
    """
    Performs constrained optimization via line search. The search direction is computed using a conjugate gradient
    algorithm, which gives x = A^{-1}g, where A is a second order approximation of the constraint and g is the gradient
    of the loss function.
    """

    def __init__(
            self,
            num_slices=1):
        """

        :param cg_iters: The number of CG iterations used to calculate A^-1 g
        :param reg_coeff: A small value so that A -> A + reg*I
        :param subsample_factor: Subsampling factor to reduce samples when using "conjugate gradient. Since the
        computation time for the descent direction dominates, this can greatly reduce the overall computation time.
        :param accept_violation: whether to accept the descent step if it violates the line search condition after
        exhausting all backtracking budgets
        :return:
        """
        Serializable.quick_init(self, locals())
       
        self._num_slices = num_slices

        self._opt_fun = None
        self._target = None
        self._max_constraint_val = None
        self._constraint_name = None
        
    def update_opt(self, loss, d_loss, target, inputs, *args,
                   **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs, which could be subsampled if needed. It is assumed
        that the first dimension of these inputs should correspond to the number of data points
        :param extra_inputs: A list of symbolic variables as extra inputs which should not be subsampled
        :return: No return value.
        """
        
        inputs = tuple(inputs)
        extra_inputs = tuple()
        params = target.get_params(trainable=True)
        grads = theano.grad(d_loss, wrt=params, disconnected_inputs='warn')
        flat_grad = ext.flatten_tensor_variables(grads)

        self.first_step=1
        self._target = target
        
        self._opt_fun = ext.lazydict(
            f_loss=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=loss,
                log_name="f_loss",
            ),
            f_grad=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=flat_grad,
                log_name="f_grad",
            ),
        )

    def loss(self, inputs):
        inputs = tuple(inputs)
        extra_inputs = tuple()
        return sliced_fun(self._opt_fun["f_loss"], self._num_slices)(inputs, extra_inputs)
    
    def optimize(self, inputs, extra_inputs=None, subsample_grouped_inputs=None):

        inputs = tuple(inputs)
        extra_inputs = tuple()
        
        logger.log("computing loss before")
        loss_before = sliced_fun(self._opt_fun["f_loss"], self._num_slices)(
            inputs, extra_inputs)
        logger.log("performing update")
        
        logger.log("computing steepest direction")
        
        
        
        flat_g = sliced_fun(self._opt_fun["f_grad"], self._num_slices)(
            inputs, extra_inputs)
        
        if self.first_step==1:
            self.conjugate_direction=flat_g
            self.dxdx=flat_g.dot(flat_g)
            self.first_step=0
        else:
            dxdx=flat_g.dot(flat_g)
            beta=dxdx/self.dxdx
            self.conjugate_direction=flat_g+beta*self.conjugate_direction
            self.dxdx=dxdx
        
        normalized_con_direction=self.conjugate_direction/np.sqrt(self.conjugate_direction.dot(self.conjugate_direction))
        
        def myfprime(cur_param):
            self._target.set_param_values(cur_param, trainable=True)
            return sliced_fun(self._opt_fun["f_grad"], self._num_slices)(
            inputs, extra_inputs)
        
        
        def f(cur_param):
            self._target.set_param_values(cur_param, trainable=True)
            loss = sliced_fun(self._opt_fun["f_loss"], self._num_slices)(inputs, extra_inputs)
            return loss
        
        prev_param = np.copy(self._target.get_param_values(trainable=True))
        alpha_star, fc, gc, phi_star, old_fval, derphi_star = scipy.optimize.line_search(f, myfprime, prev_param,-self.conjugate_direction)
        
        
        
        logger.log("optimization finished")
