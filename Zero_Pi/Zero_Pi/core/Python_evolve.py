import numpy as np
from helper_functions.grape_functions import c_to_r_mat
from helper_functions.grape_functions import c_to_r_vec
from helper_functions.grape_functions import get_state_index
import scipy.linalg as la
from scipy.special import factorial
import matplotlib.pyplot as plt

import random

from helper_functions.datamanagement import H5File

class Python_evolve:

    def __init__(self,sys_para):
        
        self.sys_para = sys_para
        self.Do_it()
        
    def approx_expm(self,M,exp_t, scaling_terms): 
        #approximate the exp at the beginning to estimate the number of taylor terms and scaling and squaring needed
        U=np.identity(len(M),dtype=M.dtype)
        Mt=np.identity(len(M),dtype=M.dtype)
        factorial=1.0 #for factorials
        
        for ii in xrange(1,exp_t):
            factorial*=ii
            Mt=np.dot(Mt,M)
            U+=Mt/((2.**float(ii*scaling_terms))*factorial) #scaling by 2**scaling_terms

        
        for ii in xrange(scaling_terms):
            U=np.dot(U,U) #squaring scaling times
        
        return U
      
    def get_norm2(self,psi):
        return np.sum(np.square(psi))
    def get_norm(self,psi):
        state_num = self.sys_para.state_num
        inter_vec_real = (psi[0:state_num])
        inter_vec_imag = (psi[state_num:2*state_num])
        inter_vec_c = inter_vec_real+1j*inter_vec_imag
        return np.dot(np.transpose(np.conjugate(inter_vec_c)),inter_vec_c)
    def evolve_no_jump(self,initial_vector):
        inter_vec = initial_vector
        inter_vecs= []
        inter_vecs.append(initial_vector)
        
        self.ops_weight_base = self.sys_para.ops_weight_base
        self.weights_unpacked=[]
        self.ops_weight = np.sin(self.ops_weight_base)
        for ii in range (self.sys_para.ops_len):
            self.weights_unpacked.append(self.sys_para.ops_max_amp[ii]*self.ops_weight[ii,:])
        for ii in np.arange(0,self.sys_para.steps):
            psi = inter_vec   
            H= self.sys_para.H0
            

            
            for jj in range (len(self.sys_para.ops)):
                H = H + self.sys_para.ops[jj]*self.weights_unpacked[jj][ii]
            U = self.approx_expm(H,self.sys_para.exp_terms,0)
            inter_vec = np.dot(U,psi)
            
            
        new_norm = self.get_norm2(inter_vec)
        inter_vec = inter_vec/np.sqrt(new_norm)
        state_num = self.sys_para.state_num
        inter_vec_real = np.asarray(inter_vec[0:state_num])
        inter_vec_imag = np.asarray(inter_vec[state_num:2*state_num])
        inter_vec_c = inter_vec_real+1j*inter_vec_imag
        loss = 1 - np.square(np.abs(np.dot(np.conjugate(np.transpose(inter_vec_c)),[0,1])))
        return new_norm, loss
    
    def Do_it(self):
        steps = self.sys_para.steps
        self.traj_num = self.sys_para.trajectories
        final_norm, no_jump_loss = self.evolve_no_jump(self.sys_para.initial_vectors[0])
        print "No jump loss: " + str(no_jump_loss)
        self.l=0
        self.l = self.l + np.rint(final_norm* self.traj_num)  * no_jump_loss / (self.traj_num)
        #print self.l
        needed_traj = int(self.traj_num-np.rint(final_norm* self.traj_num))
        self.loss_list1 = no_jump_loss * np.ones(int(self.traj_num - np.sum(needed_traj)))
        start = final_norm
        taylor_terms = self.sys_para.exp_terms 
        scaling = self.sys_para.scaling
        
        self.c_ops = np.reshape(self.sys_para.c_ops_real,[len(self.sys_para.c_ops),2*self.sys_para.state_num,2*self.sys_para.state_num])
        self.ops_weight_base = self.sys_para.ops_weight_base
        self.weights_unpacked=[]

        self.ops_weight = np.sin(self.ops_weight_base)
        for ii in range (self.sys_para.ops_len):
            self.weights_unpacked.append(self.sys_para.ops_max_amp[ii]*self.ops_weight[ii,:])

        #print len(self.sys_para.ops_max_amp)
        self.norms = []
        
       
        inter_vecs=[]
        norms = []
        print "needed traj: " + str(needed_traj)
        for initial_vector in self.sys_para.initial_vectors:
            for traj in range(needed_traj):
                total_norm = 1.0
                self.r = random.uniform(start, 1)
                
                #print "random: " + str(self.r)
                inter_vecs=[]
                inter_vec = initial_vector
                inter_vecs.append(initial_vector)
                for ii in np.arange(0,self.sys_para.steps):
                    psi = inter_vec   
                    H= self.sys_para.H0
                    norms.append(total_norm)
                    for jj in range (len(self.sys_para.ops)):
                        H = H + self.sys_para.ops[jj]*self.weights_unpacked[jj][ii]
                    U = self.approx_expm(H,taylor_terms,scaling)
                    inter_vec = np.dot(U,psi)
                    new_norm = self.get_norm2(inter_vec)
                    total_norm = total_norm*new_norm
                    inter_vec = inter_vec/np.sqrt(new_norm)
                    self.norms.append(total_norm)
                    
                    
                    if total_norm <= self.r:
                        self.r = random.uniform(0, 1)
                        #print "second random: " + str(self.r)
                        total_norm = 1
                        inter_vec = np.dot(self.sys_para.c_ops_real[0],inter_vec)
                        temp_norm = self.get_norm2(inter_vec)
                        #print "norm_after_jump: " + str(temp_norm)
                        
                        inter_vec = inter_vec/np.sqrt(temp_norm)
                        self.norms.append(total_norm)
                        
                        #ii = ii+1
                    inter_vecs.append(inter_vec)

                        

                    
                state_num = self.sys_para.state_num
                inter_vec_real = np.asarray(inter_vec[0:state_num])
                inter_vec_imag = np.asarray(inter_vec[state_num:2*state_num])
                inter_vec_c = inter_vec_real+1j*inter_vec_imag
                loss = 1 - np.square(np.abs(np.dot(np.conjugate(np.transpose(inter_vec_c)),[0,1])))
                #print "loss: " + str(loss)
                if np.isnan(loss):
                    break
                self.l = self.l + (loss/(self.sys_para.trajectories)) 
                print "loss: " + str(loss)
                self.loss_list1 = np.concatenate([self.loss_list1,[loss]])
            
            print "final loss: " + str(self.l)
            
            
            
            tlist = np.linspace(0,self.sys_para.total_time,self.sys_para.steps+1)

            np.save("python_norms",norms)

            state_num = self.sys_para.state_num
            self.inter_vecs_mag_squared = []
            ii=0
            inter_vecs = np.transpose(inter_vecs)
            shape = np.shape(inter_vecs)
            #print np.shape(inter_vecs)

            
            inter_vec_real = np.asarray(inter_vecs[0:state_num])
            inter_vec_imag = np.asarray(inter_vecs[state_num:2*state_num])
            inter_vec_c = inter_vec_real+1j*inter_vec_imag


            inter_vec_mag_squared = np.square(np.abs(inter_vec_c))






                
            #print "norm" + str(self.norms)
            np.save('python_vecs',inter_vec_mag_squared)
            #print "Before weights: "+str(np.sum(np.square(self.grad_pack)))
            fig, ax = plt.subplots(figsize=(9,6))
            ax.plot(tlist, inter_vec_mag_squared[0], label='initial')
            ax.plot(tlist, inter_vec_mag_squared[1], label='target')
            ax.legend()
            #ax.set_ylim(-0.01,1.1)
            ax.set_xlabel('Time [ns]')
            ax.set_ylabel('Occupation probability')
            #print self.norms
            
            
            
                
                
            