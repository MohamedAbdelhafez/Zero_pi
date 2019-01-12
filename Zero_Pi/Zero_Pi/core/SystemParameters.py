import numpy as np
from helper_functions.grape_functions import c_to_r_mat
from helper_functions.grape_functions import c_to_r_vec
from helper_functions.grape_functions import get_state_index
import scipy.linalg as la
from scipy.special import factorial

from helper_functions.datamanagement import H5File

class SystemParameters:

    def __init__(self,H0,Hops,Hnames,U,U0,total_time,steps,states_concerned_list,dressed_info,maxA, draw,initial_guess, show_plots,Unitary_error,state_transfer,no_scaling,reg_coeffs, save, file_path, Taylor_terms,use_gpu,use_inter_vecs,sparse_H,
                sparse_U,sparse_K, c_ops,trajectories, do_all, expect_op):
        # Input variable
        self.sparse_U = sparse_U
        self.sparse_H = sparse_H
        self.sparse_K = sparse_K
        self.use_inter_vecs = use_inter_vecs
        self.use_gpu = use_gpu
        self.Taylor_terms = Taylor_terms
        self.dressed_info = dressed_info
        self.reg_coeffs = reg_coeffs
        self.file_path = file_path
        self.state_transfer = state_transfer
        self.no_scaling = no_scaling
        self.save = save
        self.H0_c = H0
        self.ops_c = Hops
        self.ops_max_amp = maxA
        self.Hnames = Hnames
        self.Hnames_original = Hnames #because we might rearrange them later if we have different timescales 
        self.total_time = total_time
        self.steps = steps
        self.show_plots = show_plots
        self.Unitary_error= Unitary_error
        self.trajectories = trajectories
        self.c_ops = c_ops
        self.traj = False
        self.do_all = do_all
        self.expect_op = expect_op
        self.U = U
        self.expect = False
        if self.expect_op != []:
            self.expect = True


        if initial_guess is not None:
            # transform initial_guess to its corresponding base value
            self.u0 = initial_guess
            self.u0_base = np.zeros_like(self.u0)
            for ii in range (len(self.u0_base)):
                self.u0_base[ii]= self.u0[ii]/self.ops_max_amp[ii]
                if max(self.u0_base[ii])> 1.0:
                    raise ValueError('Initial guess has strength > max_amp for op %d' % (ii) )
            self.u0_base = np.arcsin(self.u0_base) #because we take the sin of weights later
                
                

            
        else:
            self.u0 =[]
        self.states_concerned_list = states_concerned_list

        self.is_dressed = False
        self.U0_c = U0
        self.initial_unitary = c_to_r_mat(U0) #CtoRMat is converting complex matrices to their equivalent real (double the size) matrices
        if self.expect:
            self.expect_op = c_to_r_mat(self.expect_op)
        
        if draw is not None:
            self.draw_list = draw[0]
            self.draw_names = draw[1]
        else:
            self.draw_list = []
            self.draw_names = []
        
        
        if dressed_info !=None:
            self.v_c = dressed_info['eigenvectors']
            self.dressed_id = dressed_info['dressed_id']
            self.w_c = dressed_info['eigenvalues']
            self.is_dressed = dressed_info['is_dressed']
            self.H0_diag=np.diag(self.w_c)
            
        self.init_system()
        self.init_vectors()
        
        if self.c_ops !=None:
            self.traj = True
            self.state_transfer = True
            
            if len(U) != len(states_concerned_list):
                full_U = U
                U=[]
                for ii in range(len(states_concerned_list)):
                    U.append(np.dot(full_U,self.initial_vectors_c[ii]))
        
        if self.state_transfer == False:
            self.target_unitary = c_to_r_mat(U)
        else:
            
            self.target_vectors=[]
            self.target_vectors_c=[]

            for target_vector_c in U:
                self.target_vector = c_to_r_vec(target_vector_c)
                self.target_vectors.append(self.target_vector)
                self.target_vectors_c.append(target_vector_c)
        if self.save:
            with H5File(self.file_path) as hf:
                hf.add('target_vectors_c',data=np.array(self.target_vectors_c))  
        if self.traj:
            self.cdaggerc=[]
            self.c_ops_real=[]
            if self.save:
                with H5File(self.file_path) as hf:
                    hf.add('c_ops',data=self.c_ops)
                   
                
            beta = 0*np.identity(self.state_num)
            print beta
            
            #ceating the effective hamiltonian that describes the evolution of states if no jumps occur
            for ii in range (len(self.c_ops)):
                cdaggerc_c = np.dot(np.transpose(np.conjugate(self.c_ops[ii])),self.c_ops[ii])
                self.c_ops_real.append(c_to_r_mat(self.c_ops[ii]+beta))
                self.cdaggerc.append(c_to_r_mat(cdaggerc_c))
                c_dagger_c_beta = np.dot(np.transpose(np.conjugate(self.c_ops[ii]) + beta),(self.c_ops[ii]+beta)) + np.dot(self.c_ops[ii],beta) - np.dot(np.transpose(np.conjugate(self.c_ops[ii])),(beta))
                self.H0_c= self.H0_c + ((0-1j)/2)* ( c_dagger_c_beta)
             
            print self.c_ops_real[0]
                
        self.init_operators()
        self.init_one_minus_gaussian_envelope()
        self.init_guess()

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
    
    def approx_exp(self,M,exp_t, scaling_terms): 
        # the scaling and squaring of matrix exponential with taylor expansions
        U=1.0
        Mt=1.0
        factorial=1.0 #for factorials
        
        for ii in xrange(1,exp_t):
            factorial*=ii
            Mt=M*Mt
            U+=Mt/((2.**float(ii*scaling_terms))*factorial) #scaling by 2**scaling_terms

        
        for ii in xrange(scaling_terms):
            U=np.dot(U,U) #squaring scaling times
        
        return U
    
    def Choose_exp_terms(self, d): 
        #given our hamiltonians and a number of scaling/squaring, we determine the number of Taylor terms
        

        exp_t = 30 #maximum

        
        if self.traj:
            for ii in range (len(self.c_ops)):
                cdaggerc_c = np.dot(np.transpose(np.conjugate(self.c_ops[ii])),self.c_ops[ii])
                
                self.H0_c= self.H0_c - ((0-1j)/2)* ( cdaggerc_c)
        H=self.H0_c
        U_f = self.U0_c
        for ii in range (len(self.ops_c)):
            H = H + self.ops_max_amp[ii]*self.ops_c[ii]
        if d == 0:
            self.scaling = max(int(2*np.log2(np.max(np.abs(-(0+1j) * self.dt*H)))),0) 

        else:
            self.scaling += d

        if self.state_transfer or self.no_scaling:
            self.scaling =0
        while True:

            if len(self.H0_c) < 10:
                for ii in range (self.steps):
                    U_f = np.dot(U_f,self.approx_expm((0-1j)*self.dt*H, exp_t, self.scaling))
                Metric = np.abs(np.trace(np.dot(np.conjugate(np.transpose(U_f)), U_f)))/(self.state_num)
            else:
                max_term = np.max(np.abs(-(0+1j) * self.dt*H))
                
                Metric = 1 + self.steps *np.abs((self.approx_exp(max_term, exp_t, self.scaling) - np.exp(max_term))/np.exp(max_term))

            if exp_t == 3:
                break
            if np.abs(Metric - 1.0) < self.Unitary_error:
                exp_t = exp_t-1
            else:
                break
        
        
        if self.traj:
            for ii in range (len(self.c_ops)):
                cdaggerc_c = np.dot(np.transpose(np.conjugate(self.c_ops[ii])),self.c_ops[ii])
                
                self.H0_c= self.H0_c + ((0-1j)/2)* ( cdaggerc_c)
        return exp_t



        
    def init_system(self):
        self.dt = float(self.total_time)/self.steps        
        self.state_num= len(self.H0_c)
        
        
    def init_vectors(self):
        # initialized vectors used for propagation
        self.initial_vectors=[]
        self.initial_vectors_c=[]
        self.target_vectors=[]
        self.target_vectors_c=[]
        self.gate_f = False

        for state in self.states_concerned_list:
            if state >0:
                if self.is_dressed:
                    self.initial_vector_c= self.v_c[:,get_state_index(state,self.dressed_id)]
                else:
                    self.initial_vector_c=np.zeros(self.state_num)
                    self.initial_vector_c[state]=1

                
            else:
                self.gate_f = True
                
                
                if state ==-1:
                    
                    
                    
                    self.initial_vector_c=np.zeros(self.state_num,dtype=complex)
                    self.initial_vector_c[0]=0.88807383
                    self.initial_vector_c[1]= 0.32505758+0.32505758*(0+1j)
                    
                elif state == -2:
                    
                    
                    self.initial_vector_c=np.zeros(self.state_num,dtype=complex)
                    self.initial_vector_c[0]=0.88807383
                    self.initial_vector_c[1]= -0.32505758-0.32505758*(0+1j)
                elif state == -3:
                    
                    self.initial_vector_c=np.zeros(self.state_num,dtype=complex)
                    self.initial_vector_c[0]=0.45970084
                    self.initial_vector_c[1]= -0.62796303+0.62796303*(0+1j)
                elif state == -4:
                    
                    
                    self.initial_vector_c=np.zeros(self.state_num,dtype=complex)
                    self.initial_vector_c[0]=0.45970084
                    self.initial_vector_c[1]= 0.62796303-0.62796303*(0+1j)
                    
            self.initial_vectors_c.append(self.initial_vector_c)
            self.initial_vector = c_to_r_vec(self.initial_vector_c)

            self.initial_vectors.append(self.initial_vector)

        
        if self.save:
            with H5File(self.file_path) as hf:
                hf.add('initial_vectors_c',data=np.array(self.initial_vectors_c))


    def init_operators(self):
        # Create operator matrix in numpy array

        self.ops=[]
        for op_c in self.ops_c:
            op = c_to_r_mat(-1j*self.dt*op_c)
            self.ops.append(op)
        
        self.ops_len = len(self.ops)

        self.H0 = c_to_r_mat(-1j*self.dt*self.H0_c)
        self.identity_c = np.identity(self.state_num)
        self.identity = c_to_r_mat(self.identity_c)
        
        if self.Taylor_terms is None:
            self.exps =[]
            self.scalings = []
            if self.state_transfer or self.no_scaling:
                comparisons = 1
            else:
                comparisons = 6
            d = 0
            while comparisons >0:

                self.exp_terms = self.Choose_exp_terms(d)
                self.exps.append(self.exp_terms)
                self.scalings.append(self.scaling)
                comparisons = comparisons -1
                d = d+1
            self.complexities = np.add(self.exps,self.scalings)
            a = np.argmin(self.complexities)

            self.exp_terms = self.exps[a]
            self.scaling = self.scalings[a]
        else:
            self.exp_terms = self.Taylor_terms[0]
            self.scaling = self.Taylor_terms[1]
            
        
        if self.save:
            with H5File(self.file_path) as hf:
                hf.add('taylor_terms',data=self.exp_terms)
                hf.add('taylor_scaling',data=self.scaling)
        
        print "Using "+ str(self.exp_terms) + " Taylor terms and "+ str(self.scaling)+" Scaling & Squaring terms"
        
        i_array = np.eye(2*self.state_num)
        op_matrix_I=i_array.tolist()
        
        self.H_ops = []
        for op in self.ops:
            self.H_ops.append(op)
        self.matrix_list = [self.H0]
        for ii in range(self.ops_len):
            self.matrix_list = self.matrix_list + [self.H_ops[ii]]
        self.matrix_list = self.matrix_list + [op_matrix_I]
        
        self.matrix_list = np.array(self.matrix_list)
        
    def init_one_minus_gaussian_envelope(self):
        # Generating the Gaussian envelope that pulses should obey
        one_minus_gauss = []
        offset = 0.0
        overall_offset = 0.01
        opsnum=self.ops_len
        for ii in range(opsnum):
            constraint_shape = np.ones(self.steps)- self.gaussian(np.linspace(-2,2,self.steps)) - offset
            constraint_shape = constraint_shape * (constraint_shape>0)
            constraint_shape = constraint_shape + overall_offset* np.ones(self.steps)
            one_minus_gauss.append(constraint_shape)


        self.one_minus_gauss = np.array(one_minus_gauss)


    def gaussian(self,x, mu = 0. , sig = 1. ):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def init_guess(self):
        # initail guess for control field
        if self.u0 != []:
            
            self.ops_weight_base = np.reshape(self.u0_base, [self.ops_len,self.steps])
            
        else:
            initial_mean = 0
            index = 0
            
            initial_stddev = (1./np.sqrt(self.steps))
            self.ops_weight_base = np.random.normal(initial_mean, initial_stddev, [self.ops_len ,self.steps])
            print np.shape(self.ops_weight_base)
        
        self.raw_shape = np.shape(self.ops_weight_base)
        
        
