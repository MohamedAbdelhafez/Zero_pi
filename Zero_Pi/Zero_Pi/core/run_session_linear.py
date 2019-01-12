import numpy as np
import tensorflow as tf
from Analysis import Analysis
import os
import time
from scipy.optimize import minimize
from helper_functions.grape_functions import *
import matplotlib.pyplot as plt
import sys

from helper_functions.datamanagement import H5File


class run_session:
    def __init__(self, tfs,graph,conv,sys_para,method,show_plots=True,single_simulation = False,use_gpu =True):
        self.tfs=tfs
        self.graph = graph
        self.conv = conv
        self.sys_para = sys_para
        self.update_step = conv.update_step
        self.iterations = 0
        self.method = method.upper()
        self.show_plots = show_plots
        self.target = False
        gpu_options = tf.GPUOptions(allow_growth=True)
        if not use_gpu:
            config = tf.ConfigProto(device_count = {'GPU': 0})
        else:
            config = tf.ConfigProto(gpu_options = gpu_options)
        
        with tf.Session(graph=graph, config = config) as self.session:
            
            tf.global_variables_initializer().run()

            print "Initialized"
            
            if self.method == 'EVOLVE':
                self.start_time = time.time()
                x0 = self.sys_para.ops_weight_base
                self.l,self.rl,self.grads,self.metric,self.g_squared=self.get_error(x0)
                self.get_end_results()
                
            else:
                if self.method != 'ADAM': #Any BFGS scheme
                    self.bfgs_optimize(method=self.method)

                if self.method =='ADAM':
                    self.start_adam_optimizer()    
                
                  
    def create_tf_vectors(self,vec_traj):
        tf_initial_vectors = []
        jj = 0
        for initial_vector in self.sys_para.initial_vectors:
            
            tf_initial_vector = np.array(initial_vector)
            for ii in range (vec_traj[jj]):
                tf_initial_vectors.append(tf_initial_vector)
            jj = jj + 1
        #tf_initial_vectors = np.transpose((tf_initial_vectors))
        
        return tf_initial_vectors
    
    def create_target_vectors(self,vec_traj):
        target_vectors = []
        jj = 0
        for target_vector in self.sys_para.target_vectors:
            
            tf_target_vector = np.array(target_vector)
            for ii in range (vec_traj[jj]):
                target_vectors.append(tf_target_vector)
            jj = jj + 1
        #target_vectors = np.transpose((target_vectors))
        
        return target_vectors
    
    def divide(self, needed,max_traj):
        returned = []
        
        end = False
        
        while not end:
            summation = 0
            trial = np.zeros_like(needed)
            flag = True

            for ii in range (len(needed)):
                if flag and ((summation + needed[ii]) <= max_traj):
                    summation = summation + needed[ii]
                    trial[ii] = needed[ii]
                    if ii == len(needed)-1:
                        end = True
                else:

                    trial[ii] = max_traj - summation
                    summation = max_traj
                    flag = False
            returned.append(trial)
            needed = needed-trial
        return returned
           
            
    def start_adam_optimizer(self):
        # adam optimizer  
        self.start_time = time.time()
        self.do_all = self.sys_para.do_all
        self.end = False
        self.cc = 0
        while True:
            
            learning_rate = float(self.conv.rate) * np.exp(-float(self.iterations) / self.conv.learning_rate_decay)
            
            
            if self.sys_para.traj and self.sys_para.expect and self.do_all:
                #Do all trajectories together
                num_psi0 = len(self.sys_para.initial_vectors)
                vec_trajs = self.sys_para.trajectories * np.ones([num_psi0])
                self.j = 0
                self.feed_dict = {self.tfs.learning_rate: 0, self.tfs.start: (np.zeros([num_psi0])), self.tfs.end: (np.ones([num_psi0])),self.tfs.num_trajs:vec_trajs}
                self.rl,self.grad_pack, self.norms,self.l,inter_vecs,all_norms, expects, linear, lineard, quad,expects2,s_g,var_g = self.session.run(
                [self.tfs.reg_loss,self.tfs.grad_pack,self.tfs.norms,self.tfs.loss,self.tfs.inter_vecs,self.tfs.all_norms, self.tfs.expectations, self.tfs.Il,self.tfs.Ild,self.tfs.quad,self.tfs.all_expectations,self.tfs.s_g,self.tfs.var_g ], feed_dict=self.feed_dict)
                #print np.shape(self.grad_pack)
                #print np.shape(lineard)
                #print (np.max(np.abs((-2*linear*lineard - quad )/quad)), np.mean(np.abs((-2*linear*lineard - quad )/quad)))
                #print (-2*linear*lineard)
                #print (quad)
                #print (-2*linear*lineard)
                #print (self.grad_pack[0])
                np.save("/home/mohamed/Data/s_g",s_g)
                np.save("/home/mohamed/Data/v_g",var_g)
                self.inter_vecs = inter_vecs
                
                self.expects = expects
                np.save("/home/mohamed/Data/expects"+str(self.cc),expects2)
                self.cc = self.cc+1
                self.occupations = self.create_avg_inter_vecs_all()
                self.feed_dict = {self.tfs.learning_rate: learning_rate, self.tfs.start: (np.zeros([num_psi0])), self.tfs.end: (np.zeros([num_psi0])),self.tfs.num_trajs:np.ones([num_psi0]), self.tfs.avg_grad: self.grad_pack}
                _ = self.session.run([self.tfs.optimizer],feed_dict = self.feed_dict)
                self.g_squared,self.metric, final, target= self.session.run([self.tfs.grad_squared,self.tfs.unitary_scale,self.tfs.final_state,self.tfs.target_vecs], feed_dict=self.feed_dict)
                
                self.grad_squared = self.g_squared
            if self.sys_para.traj and not self.do_all:
                self.traj_num = self.sys_para.trajectories
                max_traj = int(250000*250/(self.sys_para.steps * len(self.sys_para.H0)))
                #max_traj = int(250000*250/(2*self.sys_para.steps * len(self.sys_para.H0)))
                #max_traj = 1
                
                num_psi0 = len(self.sys_para.initial_vectors)
                needed_traj = np.zeros([num_psi0])
                start = (np.zeros([num_psi0]))
                end  = (np.zeros([num_psi0]))
                self.l = 0
                self.ls = [0,0]
                self.gs = []
                self.expects_no_jump = []
                self.expects_jump = []
                self.inter_vecs_no_jump = []
                self.inter_vecs_jump = []
                vec_trajs = np.ones([num_psi0])
                
                self.feed_dict = {self.tfs.learning_rate: 0, self.tfs.start: start, self.tfs.end: end,self.tfs.num_trajs:vec_trajs, self.tfs.get_reg_only: 0.0}
                self.reg_grad = np.reshape(self.session.run(
                [self.tfs.grad_pack], feed_dict=self.feed_dict),[1,1,self.sys_para.steps])
                for kk in range (num_psi0):
                    vec_trajs = np.zeros([num_psi0])
                    vec_trajs[kk] = 1
                    self.feed_dict = {self.tfs.learning_rate: 0, self.tfs.start: start, self.tfs.end: end,self.tfs.num_trajs:vec_trajs}
                    if not self.sys_para.expect:
                        self.grad_pack, self.norms,self.loss,inter_vecs,all_norms = self.session.run(
                [self.tfs.grad_pack,self.tfs.norms,self.tfs.loss,self.tfs.inter_vecs,self.tfs.all_norms], feed_dict=self.feed_dict)
                    #print (all_norms)
                        self.inter_vecs_no_jump.append(inter_vecs)
                    else:
                        
                        self.norms, expects, l1d,l2d,  quad, l1, l2, inter_vecs = self.session.run(
                [self.tfs.norms, self.tfs.expectations, self.tfs.Il1d, self.tfs.Il2d,self.tfs.quad, self.tfs.Il1, self.tfs.Il2, self.tfs.inter_vecs], feed_dict=self.feed_dict)
                        self.inter_vecs_no_jump.append(inter_vecs)
                        self.expects_no_jump.append(expects[:,kk,0] * self.norms )
                        self.grad_pack = quad
                        
                          
                    
                    
                    

                    if kk ==0:
                        self.grad_av = np.rint(self.norms* self.traj_num)  * self.grad_pack / (num_psi0*self.traj_num)
                    else:
                        self.grad_av = self.grad_av + (self.norms * self.grad_pack / num_psi0)
                    
                        
                    needed_traj[kk] = self.traj_num-np.rint(self.norms* self.traj_num) 
                    start[kk] = self.norms
                    end[kk] = 1.0
                    
                    
                self.needed = needed_traj
                jump_traj = np.sum(needed_traj)
                if jump_traj > 0 and self.sys_para.expect:
                    e0 = np.zeros(np.shape(expects[:,0,0]))
                    e1 = np.zeros(np.shape(expects[:,0,0]))
                    self.divided_branches = self.divide(needed_traj,max_traj)
                    self.time = time.time()
                    for ii in range (len(self.divided_branches)):
                        
                        #sys.stdout.write('\r'+' Iteration: ' +str(self.iterations) + ": Running batch #" +str(ii+1)+" out of "+str(len(self.divided_branches))+ " with "+str(self.divided_branches[ii])+" jump trajectories, time for last batch is "+str(time.time()-self.time))
                        self.time = time.time()
                        #sys.stdout.flush()
                        needed_traj = self.divided_branches[ii]
                        
                        #print "Doing " + str(needed_traj) + " jump trajectories with start: " + str(start)+ " and end: " + str(end)
                          
                        self.feed_dict = {self.tfs.learning_rate: 0, self.tfs.start: start, self.tfs.end: end,self.tfs.num_trajs:needed_traj}
                        if self.sys_para.expect:
                            self.rl,self.grad_pack, self.norms,self.l,inter_vecs,all_norms, expects, linear,  quad, l1, l2, l1d, l2d = self.session.run(
                [self.tfs.reg_loss,self.tfs.grad_pack,self.tfs.norms,self.tfs.loss,self.tfs.inter_vecs,self.tfs.all_norms, self.tfs.expectations, self.tfs.Il,self.tfs.quad, self.tfs.Il1, self.tfs.Il2, self.tfs.Il1d, self.tfs.Il2d], feed_dict=self.feed_dict)
                            self.others = self.rl - self.l
                            
                            
                            
                            
                            if needed_traj[0] > 0 :
                                self.ls[0] = self.ls[0] + (needed_traj[0]/(self.sys_para.trajectories)) * l1
                                self.grad_av = self.grad_av + (needed_traj[0]/(self.sys_para.trajectories)) * self.grad_pack
                                e0 = e0 + needed_traj[0]*expects[:,0,0]/(self.sys_para.trajectories)
                            if needed_traj[1] > 0 :
                                self.ls[1] = self.ls[1] + (needed_traj[1]/(self.sys_para.trajectories)) * l2
                                self.grad_av = self.grad_av + (needed_traj[1]/(self.sys_para.trajectories)) * self.grad_pack
                                e1 = e1 + needed_traj[1]*expects[:,1,0]/(self.sys_para.trajectories)
                            
                            if ii == 0:
                                self.inter_vecs_jump.append(inter_vecs)
                                
                            else:
                                inter_vecs = np.concatenate((self.inter_vecs_jump[0],inter_vecs), axis = 2)
                                self.inter_vecs_jump = []
                                self.inter_vecs_jump.append(inter_vecs)
                                
                                
                        if not self.sys_para.expect:
                            self.grad_pack,l,self.r, self.js,self.loss_list, inter_vecs = self.session.run([self.tfs.grad_pack,self.tfs.loss,self.tfs.r,self.tfs.all_jumps,self.tfs.loss_list, self.tfs.inter_vecs], feed_dict = self.feed_dict)

                            self.loss_list = np.ones(len(self.loss_list)) - self.loss_list
                            avg_jump_loss = np.mean(self.loss_list)

                            self.l = self.l + (np.sum(needed_traj)/(num_psi0*self.sys_para.trajectories)) * avg_jump_loss
                            self.grad_av = self.grad_av + (np.sum(needed_traj)/(num_psi0*self.sys_para.trajectories)) * self.grad_pack
                            if ii == 0:
                                self.inter_vecs_jump.append(inter_vecs)
                            else:
                                inter_vecs = np.concatenate((self.inter_vecs_jump[0],inter_vecs), axis = 2)
                                self.inter_vecs_jump = []
                                self.inter_vecs_jump.append(inter_vecs)
                    
                #print "After Jumps: "+str(np.sum(np.square(self.grad_av)))
                if self.sys_para.expect:
                    self.expects_jump.append(e0)
                    self.expects_jump.append(e1)
                    self.l = np.square(self.ls[0] + self.ls[1])
                    #self.grad_av = np.reshape(-2 * self.ls[0] * self.gs[0] -2 * self.ls[1] * self.gs[1] , np.shape(self.grad_pack)) + np.reshape(self.reg_grad, np.shape(self.grad_pack))
                    self.rl = self.l + self.others
                
                learning_rate = float(self.conv.rate) * np.exp(-float(self.iterations) / self.conv.learning_rate_decay)
            #print "statistical error: " + str((np.std(self.loss_list1)/np.sqrt(self.traj_num))/self.l)
                self.grad_av = np.reshape(self.grad_av, [1,1,self.sys_para.steps])
                self.feed_dict = {self.tfs.learning_rate: learning_rate, self.tfs.start: (np.zeros([num_psi0])), self.tfs.end: (np.zeros([num_psi0])),self.tfs.num_trajs:np.ones([num_psi0]), self.tfs.avg_grad: self.grad_av}
                _ = self.session.run([self.tfs.optimizer],feed_dict = self.feed_dict)

                l,self.rl,self.g_squared,self.metric, final, target= self.session.run([self.tfs.loss, self.tfs.reg_loss,self.tfs.grad_squared,self.tfs.unitary_scale,self.tfs.final_state,self.tfs.target_vecs], feed_dict=self.feed_dict)

                self.feed_dict = {self.tfs.learning_rate: 0, self.tfs.start: (np.zeros([num_psi0])), self.tfs.end: (np.zeros([num_psi0])),self.tfs.num_trajs:np.ones([num_psi0]), self.tfs.avg_grad: self.grad_av}
                
               
                self.j = jump_traj
                #self.grad_squared = np.sum(np.square(self.grad_av))
                self.grad_squared = -1
            
            if not self.sys_para.traj: 
                self.feed_dict = {self.tfs.learning_rate: learning_rate}

                self.g_squared, self.l, self.rl, self.metric = self.session.run(
                [self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss, self.tfs.unitary_scale], feed_dict=self.feed_dict)
                
            if (np.abs(self.l) < self.conv.conv_target) or (self.g_squared < self.conv.min_grad) \
                    or (self.iterations >= self.conv.max_iterations):
                self.end = True

            self.update_and_save()
                
            if self.end:
                self.get_end_results()
                break
                
            
            

            _ = self.session.run([self.tfs.optimizer], feed_dict=self.feed_dict)

                
                

    def create_avg_inter_vecs(self):
        
        state_num = int(self.sys_para.state_num)
        inter_vecs_mag_squared_list = []
        
        inter_vecs_real = []
        inter_vecs_imag = []
        
        if self.sys_para.is_dressed:
            v_sorted=sort_ev(self.sys_para.v_c,self.sys_para.dressed_id)
        
        
        result = []
        idx = 0
        for kk in range (len(self.sys_para.initial_vectors)):
            
            inter_vec_real = (self.inter_vecs_no_jump[kk][ 0:state_num,:,0])
            inter_vec_imag = (self.inter_vecs_no_jump[kk][ state_num:2*state_num,:,0])
            inter_vec_c = inter_vec_real+1j*inter_vec_imag # (4,1000,100)
            
            inter_vec_mag_squared_no_jump = np.square(np.abs(inter_vec_c)) * (1-(self.needed[kk]/self.traj_num) )
            
            inter_vec_real = (self.inter_vecs_jump[0][0:state_num,:,idx: idx + int(self.needed[kk])])
            inter_vec_imag = (self.inter_vecs_jump[0][state_num:2*state_num,:,idx: idx + int(self.needed[kk])])
            inter_vec_c = inter_vec_real+1j*inter_vec_imag
            inter_vec_mag_squared_jump = np.mean(np.square(np.abs(inter_vec_c)), axis = -1)*(self.needed[kk]/self.traj_num)

            
           
                        
            
            inter_vecs_mag_squared_list.append(inter_vec_mag_squared_jump + inter_vec_mag_squared_no_jump)
            idx = idx + int(self.needed[kk])
        
        return inter_vecs_mag_squared_list
    def create_avg_expects(self):
        
        
        
        return [x + y for x, y in zip(self.expects_no_jump, self.expects_jump)]  
    def create_avg_inter_vecs_all(self):
        state_num = self.sys_para.state_num
        inter_vecs_mag_squared_list = []
        
        inter_vecs_real = []
        inter_vecs_imag = []
        
        if self.sys_para.is_dressed:
            v_sorted=sort_ev(self.sys_para.v_c,self.sys_para.dressed_id)
        
        #inter_vecs = np.swapaxes(self.inter_vecs,0,1)
        #inter_vecs = np.swapaxes(inter_vecs,1,2)
        #(8,1000,200) 
        result = []
        idx = 0
        for kk in range (len(self.sys_para.initial_vectors)):
            
            inter_vec_real = (self.inter_vecs[0:state_num,:,idx: idx + self.sys_para.trajectories])
            inter_vec_imag = (self.inter_vecs[state_num:2*state_num,:,idx: idx + self.sys_para.trajectories])
            inter_vec_c = inter_vec_real+1j*inter_vec_imag # (4,1000,100)
            if self.sys_para.is_dressed:

                dressed_vec_c= np.dot(np.transpose(v_sorted),inter_vec_c)
                
                inter_vec_mag_squared = np.square(np.abs(dressed_vec_c))
                
                inter_vec_real = np.real(dressed_vec_c)
                inter_vec_imag = np.imag(dressed_vec_c)
                
            else:
                inter_vec_mag_squared = np.mean(np.square(np.abs(inter_vec_c)), axis = -1)
                
                inter_vec_real = np.real(inter_vec_c)
                inter_vec_imag = np.imag(inter_vec_c)
            
           
                        
            
            inter_vecs_mag_squared_list.append(inter_vec_mag_squared)
            idx = idx + self.sys_para.trajectories
        return inter_vecs_mag_squared_list
    def update_and_save(self):
        
        if not self.end:

            if (self.iterations % self.conv.update_step == 0):
                if self.do_all:
                    inters = self.create_avg_inter_vecs_all()
                    
                else:
                    inters = self.create_avg_inter_vecs()
                    if self.sys_para.expect:
                        self.expects = self.create_avg_expects()
                        self.rl = self.l + self.others
                    
                    
                self.anly = Analysis(self.sys_para, self.tfs.final_state, self.tfs.ops_weight, self.tfs.unitary_scale,
                                     self.tfs.inter_vecs,self.feed_dict, avg_inter_vecs = inters, expects = self.expects)
                self.save_data()
                self.display()
            if (self.iterations % self.conv.evol_save_step == 0):
                if not (self.sys_para.show_plots == True and (self.iterations % self.conv.update_step == 0)):
                    self.anly = Analysis(self.sys_para, self.tfs.final_state, self.tfs.ops_weight, self.tfs.unitary_scale,
                                         self.tfs.inter_vecs,self.feed_dict)
                    if not (self.iterations % self.conv.update_step == 0):
                        self.save_data()
                    self.conv.save_evol(self.anly)

            self.iterations += 1
    
    def get_end_results(self):
        # get optimized pulse and propagation
        
        # get and save inter vects
        
        self.anly = Analysis(self.sys_para, self.tfs.final_state, self.tfs.ops_weight, self.tfs.unitary_scale,
                                     self.tfs.inter_vecs,self.feed_dict)
        self.save_data()
        self.display()
        if not self.show_plots:  
            self.conv.save_evol(self.anly)
        
        self.uks = self.Get_uks()
        if not self.sys_para.state_transfer:
            self.Uf = self.anly.get_final_state()
        else:
            self.Uf = []
    
    def Get_uks(self): 
        # to get the pulse amplitudes
        uks = self.anly.get_ops_weight()
        for ii in range (len(uks)):
            uks[ii] = self.sys_para.ops_max_amp[ii]*uks[ii]
        return uks    

    def get_error(self,uks):
        #get error and gradient for scipy bfgs:
        self.session.run(self.tfs.ops_weight_base.assign(uks))

        g,l,rl,metric,g_squared = self.session.run([self.tfs.grad_pack, self.tfs.loss, self.tfs.reg_loss, self.tfs.unitary_scale, self.tfs.grad_squared])
        
        final_g = np.transpose(np.reshape(g,(len(self.sys_para.ops_c)*self.sys_para.steps)))

        return l,rl,final_g,metric, g_squared
    
    def save_data(self):
        if self.sys_para.save:
            self.elapsed = time.time() - self.start_time
            with H5File(self.sys_para.file_path) as hf:
                hf.append('error', np.array(self.l))
                hf.append('reg_error', np.array(self.rl))
                hf.append('uks', np.array(self.Get_uks()))
                hf.append('iteration', np.array(self.iterations))
                hf.append('run_time', np.array(self.elapsed))
                hf.append('unitary_scale', np.array(self.metric))
    
    
    def display(self):
        # display of simulation results

        if self.show_plots:
            self.conv.update_plot_summary(self.l, self.rl, self.anly,self.grad_squared,self.j )
        else:
            print 'Error = :%1.5e; Runtime: %.1fs; Iterations = %d, grads =  %10.3e, unitary_metric = %.5f' % (
            self.l, self.elapsed, self.iterations, self.grad_squared, self.metric) + ", jump trajectories: " + str(self.j)
    
    
    def minimize_opt_fun(self,x):
        # minimization function called by scipy in each iteration
        self.l,self.rl,self.grads,self.metric,self.g_squared=self.get_error(np.reshape(x,(len(self.sys_para.ops_c),len(x)/len(self.sys_para.ops_c))))
        
        if self.l <self.conv.conv_target :
            self.conv_time = time.time()-self.start_time
            self.conv_iter = self.iterations
            self.end = True
            print 'Target fidelity reached'
            self.grads= 0*self.grads # set zero grads to terminate the scipy optimization
        
        self.update_and_save()
        
        if self.method == 'L-BFGS-B':
            return np.float64(self.rl),np.float64(np.transpose(self.grads))
        else:
            return self.rl,np.reshape(np.transpose(self.grads),[len(np.transpose(self.grads))])

    
    def bfgs_optimize(self, method='L-BFGS-B',jac = True, options=None):
        # scipy optimizer
        self.conv.reset_convergence()
        self.first=True
        self.conv_time = 0.
        self.conv_iter=0
        self.end=False
        print "Starting " + self.method + " Optimization"
        self.start_time = time.time()
        
        x0 = self.sys_para.ops_weight_base
        options={'maxfun' : self.conv.max_iterations,'gtol': self.conv.min_grad, 'disp':False,'maxls': 40}
        
        res = minimize(self.minimize_opt_fun,x0,method=method,jac=jac,options=options)

        uks=np.reshape(res['x'],(len(self.sys_para.ops_c),len(res['x'])/len(self.sys_para.ops_c)))

        print self.method + ' optimization done'
        
        g, l,rl = self.session.run([self.tfs.grad_squared, self.tfs.loss, self.tfs.reg_loss])
            
        if self.sys_para.show_plots == False:
            print res.message
            print("Error = %1.2e" %l)
            print ("Total time is " + str(time.time() - self.start_time))
            
        self.get_end_results()          

    
        
