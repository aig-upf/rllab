from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import BatchPolopt
#from rllab.optimizers.KL_conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.optimizers.conjugate_gradient_optimizer2 import ConjugateGradientOptimizer
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer
import rllab.misc.logger as logger
import numpy as np
import theano.tensor as TT
import theano
import matplotlib.pyplot as plt
from theano import printing

import matplotlib.pyplot as plt
import rllab.misc.logger as logger

class PLAINKLANNEAL(BatchPolopt):
    """
    implementing all KL annealing methods
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            truncate_local_is_ratio=None,
            step_size=0.01,                 # epsilon for 2nd linesearch
            log_std_uncontrolled= 0,   # log_std passive dynamics
            delta = 0.2,                    # threshold 1st linesearch
            lambd = 1,                      # divides state-cost
            cg_iters = 10,
            algorithm = 'Algorithm', #which PoD to take
            PoF = 'KL_CE', #which PoF to take
            optim = 'cg',
            **kwargs
    ):
        if optimizer_args is None:
            optimizer_args = dict()
        
        optimizer = ConjugateGradientOptimizer(cg_iters=cg_iters,**optimizer_args)
        
        
        self.optimizer = optimizer
        self.step_size = step_size
        self.truncate_local_is_ratio = truncate_local_is_ratio
        self.opt_info = None
        self.log_std_uncontrolled=log_std_uncontrolled
        self.param_eta = 0.
        self.param_delta = delta
        self.f_dual = None
        self.f_opt = None
        self.lambd = lambd
        self.cg_iters = cg_iters
        self.algorithm = algorithm
        self.PoF = PoF
        super(PLAINKLANNEAL, self).__init__(optimizer=optimizer, **kwargs)
        
        # N is number of rollouts
        self.N = int(self.batch_size/self.max_path_length)
        # T is number of time-steps
        self.T = self.max_path_length
                
        logger.log("With " + str(self.N) + " rollouts per iteration")
        logger.log("With time-horizon " + str(self.T))
        logger.log('Running Natural PIREPS')
        logger.log('    delta = ' + str(self.param_delta))

    @overrides
    def init_opt(self):

        logger.log(str(self.env))
        
        ########################################################################################
        #                                                                                      #
        # This is mainly the laboratory where the symbolic functions are composed and compiled #
        #                                                                                      #
        ########################################################################################
        
        
        #########################
        # get necessary objects #
        #########################
        
        dist = self.policy.distribution #get the distribution object from the current policy
        
        ##############################################################################################################
        # Define the Theano vars which are necessary for the dual_function the statistics function and the optimizer #
        ##############################################################################################################
        X_var, U_var, V_var, mask_var, param_eta, old_dist_info_vars = self.create_theano_vars_which_serve_as_inputs(dist)


        ###########################################################################################################################
        # get the log-likelihoods of the uncontrolled dynamics and the current controller(given the inputs) (shared variables...) #
        ###########################################################################################################################
        log_pold_reshaped, log_pnew_reshaped, kl_pold_pnew, log_pold_pnew_reshaped, pnew_pold_reshaped = self.create_likelihoods_and_kl(mask_var,X_var,U_var,old_dist_info_vars,dist)

        
        
        ###############
        # create PoDs #
        ###############
        log_p_star_p_old = -(TT.sum((1/(1+param_eta))*mask_var*(V_var/self.lambd),1))
        log_p_star_p_old2 = -(TT.sum(mask_var*(V_var),1))
        
        switcher_algo = {
            'Algorithm': log_p_star_p_old,
            'trpo': log_p_star_p_old2,
            'Algorithm_antagon': -log_p_star_p_old,
        }
        
        S = switcher_algo[self.algorithm]
        
        # compute the optimal cost to go (ToDo: implement this as a thermodynamic integration)
        w = TT.exp(S - TT.max(S))
        Z_mean = TT.mean(w)
        logZ = TT.max(S) + TT.log(Z_mean)
        
        
        #############################
        # creade PoFs and gradients #
        #############################
        
        # here we create the forward (variational KL) and the backward (CE). 
        # (ToDo: may be it is possible to use thermodynamic integration to reduce the variance of the KL_CE)
        
        log_pold_pnew_reshaped_summed = TT.sum(mask_var*log_pold_pnew_reshaped,1) #summing over time
        pnew_pold_reshaped_summed = TT.exp(-log_pold_pnew_reshaped_summed) #TT.sum(mask_var*pnew_pold_reshaped,1) #summing over time
        
        log_pold_pnew_reshaped_summed_disconnected=theano.gradient.disconnected_grad(log_pold_pnew_reshaped_summed) #this makes sure that the derivative of this log-likelihood-ratio is not taken
        prop_KL_variational = -(pnew_pold_reshaped_summed*(S-logZ+log_pold_pnew_reshaped_summed_disconnected)) 
        prop_KL_variational_averaged_pure = TT.mean(prop_KL_variational)
        
        prop_KL_CE = TT.exp(S-logZ)*(S-logZ + log_pold_pnew_reshaped_summed)
        prop_KL_CE_averaged_pure = TT.mean(prop_KL_CE)
        
        
        ####
        # make versions of the cost which have a lower variance gradient
        ###
        
        prop_KL_variational = -(pnew_pold_reshaped_summed*((S-logZ+log_pold_pnew_reshaped_summed_disconnected)-TT.mean(S-logZ))) 
        prop_KL_variational_averaged = TT.mean(prop_KL_variational)/TT.std(S-logZ)
        #prop_KL_variational_averaged_pure = prop_KL_variational_averaged
        
        prop_KL_CE = (TT.exp(S-logZ)-TT.mean(TT.exp(S-logZ)))*(log_pold_pnew_reshaped_summed)
        prop_KL_CE_averaged = TT.mean(prop_KL_CE)/TT.std(TT.exp(S-logZ))
        
        prop_trpo_naive = -pnew_pold_reshaped_summed*(S-TT.mean(S))/TT.std(S)
        prop_trpo_naive_averaged = TT.mean(prop_trpo_naive)
        prop_trpo_naive_averaged_pure = TT.mean(prop_trpo_naive)
        
        #prop_trpo_naive_averaged = prop_trpo_naive_averaged_pure
        
        ##### risk seeking:       
        
        SJ=TT.exp(S-logZ)
        J=-pnew_pold_reshaped_summed*(SJ-TT.mean(SJ))/TT.std(SJ)
        
        prop_J =  TT.mean(J)
        prop_J_2 =  prop_J
        
        ##### risk averse
        prop_J_antagon =  -TT.mean(J)
        prop_J_antagon_2 =  -prop_J

                
        switcher_PoF = {
            'KL_var': tuple([prop_KL_variational_averaged_pure,prop_trpo_naive_averaged_pure]),
            'KL_CE': tuple([prop_KL_CE_averaged_pure,prop_J_2]),
            'trpo_naive': tuple([prop_trpo_naive_averaged_pure,prop_trpo_naive_averaged]),
            'J': tuple([prop_J,prop_J_2]),
            'J_antagon': tuple([prop_J_antagon,prop_J_antagon_2]),
        }
        
        PoF,PoFd = switcher_PoF[self.PoF]
        
        ################################################
        # normalized Cross entropy for the line search #
        ################################################
        
        normalized_KL_CE = (1/TT.log(self.N))*prop_KL_CE_averaged_pure #ranged between 0 (p*=pold and 1 (minimal overlap)
        
        
        #############################
        # compute some statistics  #
        #############################
        
        #basic statistics
        
        weights = (w/TT.sum(w))
        weights_max = TT.max(weights)
        weights_min = TT.mean(weights)
        
        
        S_mean = TT.mean(S)
        S_std = TT.std(S)
        
        state_cost = TT.mean(TT.sum(mask_var*(V_var),1))
        state_cost_min =  TT.min(TT.sum(mask_var*(V_var),1))
        state_cost_max =  TT.max(TT.sum(mask_var*(V_var),1))
        state_cost_std =  TT.std(TT.sum(mask_var*(V_var),1))
        
                
        total_cost = TT.mean(TT.sum(mask_var*(V_var/self.lambd),1))
        total_cost_min = TT.min(TT.sum(mask_var*(V_var/self.lambd),1))
        total_cost_max = TT.max(TT.sum(mask_var*(V_var/self.lambd),1))
        total_cost_std = TT.std(TT.sum(mask_var*(V_var/self.lambd),1))
        
                
        #ToDo:
        
        #variance measure for the gradient
        # I guess this has to be done non-symbolically somehow...
        
        #grad_deviations = (grad_PoF_unaveraged-grad_PoF)
        #grad_distances = TT.tensordot(grad_deviations,grad_deviations)
        #grad_length = TT.tensordot(grad_PoF,grad_PoF)
        
        #var_measure_gradient = grad_distances/grad_length
        
        output_statistics = [weights_max,weights_min, logZ, S_mean,S_std,state_cost,state_cost_min,state_cost_max,state_cost_std,total_cost,total_cost_min,total_cost_max,total_cost_std]
        
        #ToDo:
        
        #CE, KL, sym_KL to endgoal
        
        #eta
        
        #shortest distance to annealing curve in CE, KL, sym_KL
        
        #distance to last target (I guess that only makes sense for algorithms 2 and 3
        # for this we need to know eta_old
        
        ################################################################################
        # Compile the functions
        ################################################################################
        
        
        #########################################################################
        # dual function for the line search to find the lagrange multiplier eta #
        #########################################################################
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]
        input = [X_var, U_var, V_var, mask_var] + old_dist_info_vars_list
        input_with_eta = [X_var, U_var, V_var, mask_var] + old_dist_info_vars_list + [param_eta]
        
        self.f_dual = ext.compile_function(
            inputs=input_with_eta,
            outputs=normalized_KL_CE
        )
        
        ##########
        # function which computes some statistics
        #########

        self.f_statistics = ext.compile_function( 
            inputs=input_with_eta,         
            outputs=output_statistics
        )
        
        
        ############################
        # configure the optimizer  #
        ############################
        
        
        
        #create the input list for the optimizer
        input_list = input
        mean_kl = TT.mean(kl_pold_pnew)
        self.optimizer.update_opt(
            loss=PoF,
            d_loss=PoFd,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_with_eta,
            constraint_name="mean_kl",
        )


        
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):

        # collect data from samples
        # get effective number of rollouts
        input_values = self.collect_data(samples_data)

        #############
        # line search --> skip it if we use trpo
        #############
        if self.algorithm == 'trpo':
            param_eta = np.ones((self.N,self.T))
        else:    
            param_eta = self.compute_eta(input_values, input_values)
        
        

        #######################
        # natural PICE gradient
        #######################
        input_values_with_eta = input_values+[param_eta]
        # get some statistics to log
        loss_before = self.optimizer.loss(input_values_with_eta)

        # call optimize
        self.optimizer.optimize(input_values_with_eta)
        
        # get some statistics to log
        loss_after = self.optimizer.loss(input_values_with_eta)
        
        #####################
        # compute statistics an log them logging
        #####################
        
        #compute statistics
        weights_max,weights_min, logZ, S_mean,S_std,state_cost,state_cost_min,state_cost_max,state_cost_std,total_cost,total_cost_min,total_cost_max,total_cost_std = self.f_statistics(*input_values_with_eta)
        
        logger.record_tabular('State Cost', state_cost)
        
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('dLoss', loss_before - loss_after)
        
        logger.record_tabular('Total cost', total_cost)
        logger.record_tabular('Totalcostmin', total_cost_min)
        logger.record_tabular('Totalcostmax', total_cost_max)
        logger.record_tabular('Totalcoststd', total_cost_std)
        
        logger.record_tabular('Sstd', S_std)
        logger.record_tabular('Smean', S_mean)
        logger.record_tabular('largest weight', weights_max)
        logger.record_tabular('smallest weight', weights_min)
        
        logger.record_tabular('logZ', logZ)
        #logger.record_tabular('var_measure_gradient', var_measure_gradient)
        print("ETA")
        print(param_eta[0,0])
        logger.record_tabular('eta', param_eta[0,0]) #param_eta is a matrix with identical entries (because of the cg optimzier this has to be like this... does not take scalar inputs...)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
    
    def collect_data(self,samples_data):
        # collect data from samples
        # get effective number of rollouts
        self.N = len(samples_data["V"])
        input_values = [samples_data["X"], samples_data["U"],
                            samples_data["V"], samples_data["mask"]]
        
        print("------------ arrived " + str(self.N) + " rollouts")
        
        agent_infos2 = samples_data["agent_infos2"]
        dist_info_list = [agent_infos2[k] for k in self.policy.distribution.dist_info_keys]
        input_values_with_agent = input_values + dist_info_list
        
        
        
        return input_values_with_agent
    
    def compute_eta(self,input_values, input_values_with_agent):
        #save rel entropy
        param_eta = np.zeros((self.N,self.T))
        input_values_with_eta = input_values + [param_eta]
        
        rel_entropy = self.f_dual(*input_values_with_eta)
        logger.log("rel_entropy_before " + str(rel_entropy))

        #############
        # line search
        #############
        outer_it = 15 
        min_log = -30
        max_log = 20 
        it = 0
        
        # add zero in the beginning
        nit = 100
        rang = np.logspace(min_log,max_log,nit)
        min_eta = rang[0]
        max_eta = rang[1]
        converged = False
        while (not converged) and it<outer_it :
            veta = np.zeros(nit)
            vent = np.zeros(nit)
            i = 0
            while (i<nit) :
                #print("it = " + str(it) + " i = " + str(i))
                eta = rang[i]
                param_eta = np.ones((self.N,self.T))*eta #bring eta into a form which is possible to subsample as the extra_inputs of the cg-optimizer is buggy
                input_values_with_eta = input_values + [param_eta]
                rel_entropy = self.f_dual(*input_values_with_eta)
                final_rel_entropy = rel_entropy
                
                #print("it " + str(it) + " i " + str(i) + ": entropy " +str(rel_entropy))
                veta[i] = eta
                vent[i] = rel_entropy
                if rel_entropy < self.param_delta and i > 0:
                    #print("passed")
                    eta = rang[i]
                    final_rel_entropy = vent[i]
                    min_eta = rang[i-1]
                    max_eta = rang[i]
                    break
                i += 1
            dif = abs(self.param_delta-final_rel_entropy);
            converged = dif < 1e-5
            print("it " + str(it) + " eta " + str(eta) + str(i)
                  + ": entropy " + str(final_rel_entropy) + " diff " +
                  str(dif))
            it += 1
            nit = 10
            rang = np.linspace(min_eta,max_eta,nit)
            #print("new range " + str(min_eta) + "/" + str(max_eta))

        # check again?
        rel_entropy = final_rel_entropy
        if rel_entropy > self.param_delta : 
            logger.log("------------------ Line search for eta failed (2) !!!")

        logger.log("eta is      " + str(eta))
        logger.log("rel_entropy " + str(rel_entropy))
        
        return param_eta
    
    
    def create_theano_vars_which_serve_as_inputs(self,dist):
        # Theano vars
        X_var = ext.new_tensor(
            'X',
            ndim=2,
            dtype=theano.config.floatX,
        )   # a N*T x state_dim array 
        U_var = ext.new_tensor(
            'U',
            ndim=2,
            dtype=theano.config.floatX,
        )   # a N*T x action_dim array 
        V_var = ext.new_tensor(
            'V',
            ndim=2,
            dtype=theano.config.floatX,
        )   # corresponds to V
        
        mask_var = ext.new_tensor(
            'mask',
            ndim=2,
            dtype=theano.config.floatX,
        )   # corresponds to the mask which tells how long the rollout actually was
        
        #param_eta = TT.scalar('eta') #the lagrange multiplier
        param_eta = ext.new_tensor(
            'eta',
            ndim=2,
            dtype=theano.config.floatX,
        )
        
        weights_var = ext.new_tensor(
            'W',
            ndim=2,
            dtype=theano.config.floatX,
            # the weights w~p*/p_theta
        ) 
            
        # mean and std of the policy
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
        }
        
        
        return X_var, U_var, V_var, mask_var, param_eta, old_dist_info_vars, 
    
    def create_likelihoods_and_kl(self,mask_var, X_var,U_var,old_dist_info_vars,dist):
            #this computes the distribution of the 
        dist_info_vars = self.policy.dist_info_sym(X_var) #for this the X_var needs to be a matrix samples vs input_dimensions...
        
        log_pnew = dist.log_likelihood_sym(U_var, dist_info_vars)
        
        log_pold = dist.log_likelihood_sym(U_var, old_dist_info_vars)
        
               
                
        pnew_pold = dist.likelihood_ratio_sym(U_var, old_dist_info_vars, dist_info_vars)
        pnew_pold_reshaped = pnew_pold.reshape((self.N,self.T)) 
        
        log_pnew_reshaped = log_pnew.reshape((self.N,self.T))
        log_pold_reshaped = log_pold.reshape((self.N,self.T))
        
        kl_pold_pnew = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        
        log_pold_pnew = dist.log_likelihood_ratio_sym(U_var, dist_info_vars,old_dist_info_vars)
        log_pold_pnew_reshaped = log_pold_pnew.reshape((self.N,self.T)) 
        
        return log_pold_reshaped, log_pnew_reshaped, kl_pold_pnew, log_pold_pnew_reshaped, pnew_pold_reshaped
    