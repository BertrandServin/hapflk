#!python
#cython: language_level=3, boundscheck=False
'''
Module implementing the LD model of Scheet and Stephens (2006)
originally implemented in Paul Scheet fastPHASE software.
Code is largely inspired by Yongtao Guan Bimbam Code and as such
is released under the GNU General Public Licence version 2

Multithread version
'''
cimport cython
## NUMPY import in cython
import numpy as np
from scipy.optimize import linear_sum_assignment
from libc.math cimport log
cimport numpy as np
np.import_array()
from multiprocessing import Pool, cpu_count
import sys
from functools import reduce
from operator import add


class fitInData():
    def __init__(self,type,alpha,theta,rho,data,u2z):
        self.type=type
        self.alpha=alpha
        self.theta=theta
        self.rho=rho
        self.data=data
        self.up2pz=u2z

class fitOutData():
    def __init__(self,logLike,top,bot,jmk,val):
        self.logLike=logLike
        self.top=top
        self.bot=bot
        self.jmk=jmk
        self.val=val

class imputeInData():
    def __init__(self,type,parList,nLoc,name,data):
        self.type=type
        self.parList=parList
        self.name=name
        self.nLoc=nLoc
        self.data=data
        
class imputeOutData():
    def __init__(self,pgeno,name,pZ,path=None):
        self.name=name
        self.pgeno=pgeno
        self.pZ=pZ
        self.path=path

class modParams():
    '''
    A class for fastphase model parameters.
    Dimensions:
    -- number of loci: N
    -- number of clusters: K
    Parameters:
    -- theta (N x K): allele frequencies in clusters at each locus
    -- alpha (N x K): cluster weights at each locus
    -- rho (N x 1): jump probabilities in each interval
    '''
    def __init__(self,nLoc,nClus,rhomin=1e-6, alpha_up = True, theta_up = True):
        self.nLoc=nLoc
        self.nClus=nClus
        self.theta=0.98*np.random.random((nLoc,nClus))+0.01 # avoid bounds 0 and 1
        self.rho=np.ones((nLoc,1))/1000
        ##self.alpha=np.random.mtrand.dirichlet(np.ones(nClus),nLoc) # make sure sum(alpha_is)=1
        self.alpha=1.0/nClus*np.ones((nLoc,nClus))
        self.rhomin=rhomin
        self.alpha_up = alpha_up
        self.theta_up = theta_up
        self.loglike = 0
    def initUpdate(self):
        self.top=np.zeros((self.nLoc,self.nClus))
        self.bot=np.zeros((self.nLoc,self.nClus))
        self.jmk=np.zeros((self.nLoc,self.nClus))
        self.jm=np.zeros((self.nLoc,1))
        self.nhap=0.0
    def addIndivFit(self,t,b,j,nhap):
        self.top += t
        self.bot += b
        self.jmk += j
        self.jm  += np.reshape(np.sum(j,axis=1),(self.nLoc,1))
        self.nhap+=nhap
    def update(self):
        ''' Update parameters using top,bot,jmk jm probabilities'''
        ## rho
        ##self.rho=self.jm/self.nhap
        for i in range(self.nLoc):
            self.rho[i,0]=self.jm[i,0]/self.nhap
            if self.rho[i,0]<self.rhomin:
                self.rho[i,0]=self.rhomin
            elif self.rho[i,0]>(1-self.rhomin):
                self.rho[i,0]=1-self.rhomin
        ## alpha
        #self.alpha=self.jmk/self.jm
        if self.alpha_up:
            for i in range(self.nLoc):
                for j in range(self.nClus):
                    self.alpha[i,j]=self.jmk[i,j]/self.jm[i,0]
                    if self.alpha[i,j]>=0.999:
                        self.alpha[i,j]=0.999
                    elif self.alpha[i,j]<0.001:
                        self.alpha[i,j]=0.001
                self.alpha[i,:] /= np.sum(self.alpha[i,:])
        ## theta
        if self.theta_up:
            self.theta=self.top/self.bot
            for i in range(self.nLoc):
                for j in range(self.nClus):
                    if self.theta[i,j]>0.999:
                        self.theta[i,j]=0.999
                    elif self.theta[i,j]<0.001:
                        self.theta[i,j]=0.001
    def write(self,stream=sys.stdout):
        print("snp", *["t"+str(i) for i in range(self.nClus)], "rho", *["a"+str(i) for i in range(self.nClus)], file=stream)
        for i in range(self.nLoc):
            print(i, *[np.round( self.theta[i,k], 3) for k in range(self.nClus)], np.round( self.rho[i,0], 7), *[np.round( self.alpha[i,k], 3) for k in range(self.nClus)], file=stream)

_tohap = np.vectorize(lambda x: (x==1) and -1 or (x/2))
    
class fastphase():
    '''
    A class to manipulate and control a fastphase model (Scheet and Stephens, 2006)
    Initialized with a problem size = number of loci

    Usage : 
    with fastphase(nloc, nproc) as fph:
         ... do stuff ...
    '''
    def __init__(self, nLoci, nproc = cpu_count(),prfx=None):
        assert nLoci>0
        self.nLoci=nLoci
        self.haplotypes={}
        self.genotypes={}
        self.nproc = nproc
        self.pool = None
        self.prfx = prfx
        
    def __enter__(self):
        self.pool = Pool ( self.nproc)
        if self.prfx is None:
            self.flog = sys.stdout
        else:
            self.flog = open(self.prfx, 'w')
        return self

    def __exit__(self,*args):
        self.pool.terminate()
        if self.prfx is not None:
            self.flog.close()

    def flush(self):
        '''
        remove data 
        '''
        self.haplotypes={}
        self.genotypes={}
    
    def addHaplotype(self,ID,hap,missing=-1):
        '''
        Add an haplotype to the model observations.
        hap is a numpy array of shape (1,nLoci).
        Values must be 0,1 or missing
        '''
        try:
            assert hap.shape[0]==self.nLoci
            self.haplotypes[ID]=hap
        except AssertionError:
            print("Wrong Haplotype Size:",hap.shape[0],"is not",self.nLoci)
            raise
    def addGenotype(self,ID,gen,missing=-1):
        '''
        Add a genotype to the model observations.
        gen is a numpy array of shape (1,nLoci).
        Values must be 0,1,2 or missing
        '''
        try:
            assert gen.shape[0]==self.nLoci
            self.genotypes[ID]=gen
        except AssertionError:
            print("Wrong Genotype Size:",gen.shape[0],"is not",self.nLoci)
            raise

    @staticmethod
    def gen2hap(gen):
        return np.array( _tohap(np.array(gen, dtype=int)), dtype=int)
    
    def fit(self,nClus=20,nstep=20,params=None,verbose=False,rhomin=1e-6, alpha_up = True, theta_up = True, fast=False):
        '''
        Fit the model on observations with nCLus clusters using nstep EM iterations
        Multithread version.
        '''
        try:
            assert self.pool is not None
        except AssertionError:
            print('Usage :\n\t with fastphase(nloc, nproc) as fph: \n ...')
            raise
            
        if params:
            par=params
            par.alpha_up = alpha_up
            par.theta_up = theta_up
        else:
            par=modParams(self.nLoci,nClus,rhomin, alpha_up, theta_up)
        if verbose:
            print( 'Fitting fastphase model',file=self.flog)
            print( '# clusters ',nClus, file=self.flog)
            print( '# threads ', self.nproc, file=self.flog)
            print( '# Loci', self.nLoci, file=self.flog)
            print( '# Haplotypes',len(self.haplotypes), file=self.flog)
            print( '# Genotypes', len(self.genotypes), file=self.flog)
        old_log_like=1
  
        for iEM in range(nstep):
   
            log_like=0.0
            par.initUpdate()

            tasks =  [ fitInData( 'haplo', par.alpha,par.theta,par.rho,hap,0) for hap in  self.haplotypes.values()]
            if fast:
                tasks += [ fitInData( 'haplo', par.alpha,par.theta,par.rho,fastphase.gen2hap(gen),0) for gen in  self.genotypes.values()]
            else:
                tasks += [ fitInData( 'geno', par.alpha,par.theta,par.rho,gen,0) for gen in  self.genotypes.values()]
            results = self.pool.map( fitter, tasks)

            for item in results:
                par.addIndivFit(item.top,item.bot,item.jmk,item.val)
                log_like += item.logLike
            if verbose:
                print( iEM, log_like, file=self.flog)
                self.flog.flush()
            par.update()
            par.loglike=log_like
        
        return par

    def optimfit(self, nClus = 20, nstep = 10, params=None, verbose = False, rhomin=1e-6, alpha_up = False, fast=False, nEM = 10, niter=5):
        liktraj = []

        print("=== OPTIMFIT RUN ===",file=self.flog)
        ## Fit nEM EM to find an initial start
        if verbose:
            print("*** Init EM", 1,file=self.flog)
        nbest = 0
        curpar = self.fit( nClus = nClus, nstep = nstep, verbose = True, alpha_up = alpha_up, fast=fast)
        liktraj.append((0, 0, curpar.loglike))
        for n in range(1, nEM):
            if verbose:
                print("*** Init EM", n+1,file=self.flog)
            
            par = self.fit( nClus = nClus, nstep = nstep,verbose = True, alpha_up = alpha_up, fast=fast)
            liktraj.append((n, 0, par.loglike))
            if par.loglike > curpar.loglike:
                nbest = n
                curpar = par
        init_par = curpar
        ## Improve it
        print('Initial Viterbi',file=self.flog)
        self.flog.flush()
        imp = self.viterbi([curpar])
        
        for it in range(niter):
            if verbose:
                print("*** Iter", it+1,file=self.flog)
                self.flog.flush()
            ## calculate costs
            if verbose:
                print("Calculating Costs",file=self.flog)
                self.flog.flush()
            ##cost_mat = np.zeros( (self.nLoci-1, nClus, nClus), dtype=np.float)
            cost_mat = [ np.zeros( (nClus,nClus), dtype = np.float) for i in range(self.nLoci-1)]
            ### Genotypes
            gen_dat = ( ( imp[geno][0][ijump:ijump+2], nClus) for geno in self.genotypes.keys() for ijump in range(self.nLoci-1)  )
            for geno in self.genotypes.keys():
                vit = imp[geno][0]
                args = [(vit[ijump:ijump+2], nClus) for ijump in range(self.nLoci -1)]
                ## res is [ np.array(nK,nK) , ... ,np.array(nK,nK)]
                res = self.pool.map( calc_cost_matrix_geno, args, 100)
                ## " cost_mat += res "
                cost_mat = list( map( add, cost_mat, res))
            ### haplotypes
            for haplo in self.haplotypes.keys():
                vit = imp[haplo][0]
                args = [(vit[ijump:ijump+2], nClus) for ijump in range(self.nLoci -1)]
                ## res is [ np.array(nK,nK) , ... ,np.array(nK,nK)]
                res = self.pool.map( calc_cost_matrix_haplo, args, 100)
                ## " cost_mat += res "
                cost_mat = list( map( add, cost_mat, res))
                ##cost_mat += self.pool.map( calc_cost_matrix_haplo, args, 100)
            ## combine
            if verbose:
                print("Computing optimum permutations",file=self.flog)
                self.flog.flush()
            ##cost_mat = cost_mat.reshape( ( self.nLoci-1, nClus, nClus))
            ##args = [ cost_mat[i] for i in range( self.nLoci-1)]
            res = np.array( self.pool.map( linear_sum_assignment, cost_mat,  100))
            permut = res[:,1,:]
            newpar = switch_pars( curpar, permut)
            if verbose:
                print("EM with switched parameters",file=self.flog)
                self.flog.flush()
            if (it == niter-1): ## last iteration, longer and no imputation
                par_s = self.fit( nClus = nClus, nstep=max(30,nstep), verbose=True, params=newpar, alpha_up=False, fast=fast)
            else:
                par_s = self.fit( nClus = nClus, nstep=nstep, verbose=True, params=newpar, alpha_up=False, fast=fast)
                imp = self.viterbi([par_s])
            curpar = par_s
            liktraj.append((nbest, it+1, curpar.loglike))
        
        ## add alpha update
        # par_s = self.fit(nClus = nClus, nstep = max(30,nstep), verbose = True, params = par_s, alpha_up = True, theta_up = False, fast = fast)
        # liktraj.append((nbest, it+2, par_s.loglike))
        print('EM','iter','loglik',file=self.flog)
        for dat in liktraj:
            print(*dat,file=self.flog)
        return par_s

    def viterbi(self, parList):
        tasks =  [ imputeInData('haplo',parList,self.nLoci,name,hap) for name, hap in  self.haplotypes.items()]
        tasks += [ imputeInData('geno',parList,self.nLoci,name,gen) for name, gen in self.genotypes.items()]
        results = self.pool.map( viterber, tasks)
        Imputations={}
        for item in results:
            Imputations[item.name]=item.path
        return Imputations

    
        # results=[]
        # nblocks = len(tasks)//self.nproc
        # reste = len(tasks)%self.nproc
        # for ib in range(nblocks-1):
        #     results += self.pool.map( viterber, tasks[ib*nproc:(ib+1)*nproc])

        # Imputations={}
        # for item in results:
        #     Imputations[item.name]=item.path

        # ## Memory efficient approach, only create nproc tasks at a time
        # ib = 0
        # tasks = []
        # results = []

        # ##### Haplotyps
        # for name, hap in self.haplotypes.items():
        #     tasks.append(imputeInData('haplo',parList,self.nLoci,name,hap))
        #     ib += 1
        #     if ib == self.nproc:
        #         results += self.pool.map( viterber, tasks)
        #         ib = 0
        #         tasks = []
        # ## finish the job
        # if len(tasks) >0:
        #     results += self.pool.map( viterber, tasks)
        #     ib = 0
        #     tasks=[]
        # ##### Genotypes
        # for name, gen in self.genotypes.items():
        #     ib += 1
        #     tasks.append(imputeInData('geno',parList,self.nLoci,name,gen))
        #     if ib == self.nproc:
        #         results += self.pool.map( viterber, tasks)
        #         ib = 0
        #         tasks = []
        # ## finish the job
        # if len(tasks) >0:
        #     results += self.pool.map( viterber, tasks)
        #     ib = 0
        #     tasks=[]
    
    def impute(self,parList):
        tasks =  [ imputeInData('haplo',parList,self.nLoci,name,hap) for name, hap in  self.haplotypes.items()]
        tasks += [ imputeInData('geno',parList,self.nLoci,name,gen) for name, gen in self.genotypes.items()]
        results = self.pool.map( imputer, tasks)
        Imputations={}
        for item in results:
            Imputations[item.name]=(item.pgeno,item.pZ,item.path)
        return Imputations



def fitter( item):
    #type,alpha,theta,rho,data = item
    if item.type == 'haplo' :
        try:
            hLogLike,top,bot,jmk=hapCalc(item.alpha,item.theta,item.rho,item.data,0)
        except ValueError:
            print( item.data)
            raise
        res = fitOutData(hLogLike,top,bot,jmk,1)
    elif item.type == 'geno' :
        gLogLike,top,bot,jmk=genCalc(item.alpha,item.theta,item.rho,item.data,0)
        res = fitOutData(gLogLike,top,bot,jmk,2)
    return res

def viterber( item):
    #cIter=iter
    cdef int i,k
    cdef int nLoc
    cdef double x
    cdef double nsamp=100
    if item.type == 'haplo':
        path = []
        for par in item.parList:
            pth = hapViterbi( par.alpha, par.theta, par.rho, item.data)
            path.append(pth)
        res = imputeOutData(None,item.name,None,path)
    elif item.type == 'geno':
        path = []
        for par in item.parList:
            pth = genViterbi( par.alpha, par.theta, par.rho, item.data)
            path.append( pth)
        res = imputeOutData(None,item.name,None, path)
    return res

def imputer( item):
    #cIter=iter
    cdef int i,k
    cdef int nLoc
    cdef double x
    cdef double nsamp=100
    if item.type == 'haplo':
        pgeno=np.zeros(item.nLoc,dtype=np.float64)
        probZ=[]
        path = []
        x = 1.0/len(item.parList)
        for par in item.parList:
            pZ=hapCalc(par.alpha,par.theta,par.rho,item.data,1)
            for i in range(item.nLoc):
                for k in range(par.nClus):
                    pgeno[i]+=x*hap_p_all(item.data[i],pZ[i,k],par.theta[i,k])
            probZ.append(pZ)
            pth = None#hapViterbi( par.alpha, par.theta, par.rho, item.data)
            path.append(pth)
        res = imputeOutData(pgeno,item.name,probZ,path)
    elif item.type == 'geno':
        pgeno=np.zeros(item.nLoc,dtype=np.float64)
        probZ=[]
        path = []
        x = 1.0/len(item.parList)
        for par in item.parList:
            pZ=genCalc(par.alpha,par.theta,par.rho,item.data,1)
            ## Calculate mean genotypes
            for i in range(item.nLoc):
                for k1 in range(par.nClus):
                    for k2 in range(par.nClus):
                        pgeno[i]+=gen_p_geno(item.data[i],pZ[i,k1,k2],par.theta[i,k1],par.theta[i,k2])
            probZ.append(pZ)
            pth = None#genViterbi( par.alpha, par.theta, par.rho, item.data)
            path.append( pth)
        pgeno/=len(item.parList)
        res = imputeOutData(pgeno,item.name,probZ, path)
    return res

##### Cluster switch functions

cpdef np.ndarray[np.float64_t, ndim=3] calc_cost_matrix_haplo_tot( np.ndarray[np.int32_t, ndim=1] h, int nK ):
    cdef int l,nL
    nL= h.shape[0]
    cdef np.ndarray[np.float64_t, ndim=3] res = np.zeros( ( nL-1, nK, nK), dtype=np.float64)
    
    for l in range(nL -1):
        res[l, h[l], h[l+1]] -= 1
    return res

def calc_cost_matrix_haplo( args):
    vit, nK = args
    k = vit[0]
    kp = vit[1]
    cost_mat = np.zeros((nK,nK))
    cost_mat[k, kp] -= 1
    return cost_mat

cpdef np.ndarray[np.float64_t, ndim=3] calc_cost_matrix_geno_tot( np.ndarray[np.int32_t, ndim=2] g, int nK ):
    cdef int l,nL
    cdef int k1, k2, kp1, kp2
    nL= g.shape[0]
    cdef np.ndarray[np.float64_t, ndim=3] res = np.zeros( ( nL-1, nK, nK), dtype=np.float64)

    for l in range(nL -1):
        if (k1 == kp1):
            if (k2 == kp2): ## k1 == kp1 & k2 == kp2 , possibly k1 == k2
                res[l, k1, kp1] -= 1
                res[l, k2, kp2] -= 1
            else: ## k1 == kp1 & k2 != kp2
                res[l, k1, kp1] -= 1 ## diagonal
                res[l, k2, kp2] -= 1
        elif ( k2 == kp2): ## k1 != kp1 & k2 == kp2
            res[l, k1, kp1] -= 1 ## diagonal
            res[l, k2, kp2] -= 1
        elif ( k1 == kp2) and (k2 != kp1):
            res[l, k1, kp2] -= 1 ## diagonal
            res[l, k2, kp1] -= 1
        elif ( k1 != kp2) and (k2 == kp1):
            res[l, k1, kp2] -= 1 ## diagonal
            res[k2, kp1] -= 1
        else:
            res[l, k1, kp1] -= 0.5
            res[l, k1, kp2] -= 0.5
            res[l, k2, kp1] -= 0.5
            res[l, k2, kp2] -= 0.5
    return res

def calc_cost_matrix_geno( args):
    vit, nK = args
    k1, k2 = vit[0]
    kp1, kp2 = vit[1]
    cost_mat = np.zeros((nK,nK))
    if (k1 == kp1):
        if (k2 == kp2): ## k1 == kp1 & k2 == kp2 , possibly k1 == k2
            cost_mat[k1, kp1] -= 1
            cost_mat[k2, kp2] -= 1
        else: ## k1 == kp1 & k2 != kp2
            cost_mat[k1, kp1] -= 1 ## diagonal
            cost_mat[k2, kp2] -= 1
    elif ( k2 == kp2): ## k1 != kp1 & k2 == kp2
        cost_mat[k1, kp1] -= 1 ## diagonal
        cost_mat[k2, kp2] -= 1
    elif ( k1 == kp2) and (k2 != kp1):
        cost_mat[k1, kp2] -= 1 ## diagonal
        cost_mat[k2, kp1] -= 1
    elif ( k1 != kp2) and (k2 == kp1):
        cost_mat[k1, kp2] -= 1 ## diagonal
        cost_mat[k2, kp1] -= 1
    else:
        cost_mat[ k1, kp1] -= 0.5
        cost_mat[ k1, kp2] -= 0.5
        cost_mat[ k2, kp1] -= 0.5
        cost_mat[ k2, kp2] -= 0.5
    return cost_mat


def switch_pars(par, permut):
    newpar = modParams(par.nLoc, par.nClus, alpha_up = par.alpha_up)
    for i in range(par.nClus):
        newpar.theta[0,i]=par.theta[0,i]
        curclus = i
        for j in range(1, par.nLoc):
            curclus =  permut[ j-1, curclus]
            newpar.theta[ j,i] = par.theta[ j, curclus]
    return newpar

############################### Calculations #################################

cdef double hap_p_all(int i, double pz, double theta):
    cdef double rez
    if i<0:
        rez=pz*theta
    else:
        rez=i*pz
    return rez

cdef double gen_p_geno(int i, double pz, double theta1, double theta2):
    cdef double rez
    ## theta1*(1-theta2)+theta2*(1-theta1)+2*theta2*theta1 == theta1+theta2
    if i<0:
        rez=pz*(theta1+theta2)
    else:
        rez=i*pz
        #rez=pz*(theta1+theta2)
    return rez


cdef double myPow10(int x):
    cdef double rez=1.0
    cdef int i
    if x==0:
        return rez
    elif x<0:
        for i in range(-x):
            rez*=0.1
    else:
        for i in range(x):
            rez*=10
    return rez


cdef int argmax( np.ndarray[ np.float64_t, ndim=1] a, int size):
    cdef int i, best_i
    cdef double v, best_v

    best_v = a[ 0 ]
    best_i = 0
    for i in range(1, size):
        if a[ i ] > best_v:
            best_i = i
            best_v = a[ i ]
    return best_i

cdef int pair2idx( int k1, int k2, int nK):
    cdef int k, l
    k = min( k1, k2)
    l = max( k1, k2)
    return  nK * k - ( k * ( k -1))//2  + ( l - k)

cdef ( int, int) idx2pair( int idx, int nK):
    cdef int k, l, nelem, curk
    nelem = 0
    for curk in range(nK):
        if ( idx - nelem) < ( nK - curk):
            l = curk + (idx - nelem)
            k = curk
            break
        else:
            nelem += nK - curk
    return (k, l)


##### Genotype Calculations 

cdef double genprG(double t1, double t2, int g):
    cdef double rez
    if g==0:
        rez=(1-t1)*(1-t2)
    elif g==1:
        rez=t1+t2-2*t1*t2
    elif g==2:
        rez=t1*t2
    else:
        rez=1
    return rez

cdef double probJ(int m,int s, double rho):
    if s==0:
        return (1-rho)*(1-rho)
    elif s==1:
        return 2*(1-rho)*rho
    elif s==2:
        return rho*rho

cpdef genViterbi( aa, tt, rr, gg):
    cdef np.ndarray[np.float64_t, ndim=2] alpha = aa
    cdef np.ndarray[np.float64_t,ndim=2] theta = tt
    cdef np.ndarray[np.float64_t, ndim=2] rho = rr
    cdef np.ndarray[np.int_t, ndim=1] gen = gg

    ## cython declarations
    cdef int nLoc, nK, ikp,  prev_kp, npairs
    cdef int k1, k2, kp1, kp2, m
    cdef double best_v
    
    nLoc = alpha.shape[0]
    nK = alpha.shape[1]
    npairs = nK * ( nK + 1)//2
    
    cdef np.ndarray[ np.float64_t, ndim = 2 ] delta = np.zeros((npairs, nLoc), dtype=np.float64)
    cdef np.ndarray[ np.int_t, ndim = 2 ] psi = np.zeros((npairs, nLoc), dtype=np.int)
    cdef np.ndarray[ np.int_t, ndim = 1] soluce = np.zeros(nLoc, dtype= np.int)
    cdef np.ndarray[ np.float64_t, ndim = 1] tempVal = np.zeros( npairs, dtype=np.float64)


    ## initialization
    for k1 in range(nK):
        ikp = pair2idx( k1, k1, nK)
        delta[ikp,0] = log(alpha[ 0, k1]) + log(alpha[ 0, k1]) +log( genprG( theta[ 0, k1], theta[ 0, k1], gen[0]))
        for k2 in range( k1+1 , nK):
            ikp = pair2idx( k1, k2, nK)
            delta[ikp,0] = log(2) + log( alpha[ 0, k1]) + log( alpha[ 0, k2]) +log( genprG( theta[ 0, k1], theta[ 0, k2], gen[0]))

    ## recursion
    for m in range(1, nLoc):
        for ikp in range(npairs):
            k1, k2 = idx2pair( ikp, nK)
            for prev_kp in range(npairs):
                kp1, kp2 = idx2pair( prev_kp, nK)
                if (k1 == kp1):
                    if (k2 == kp2): ## k1 == kp1 & k2 == kp2
                        tempVal[ prev_kp] = log(probJ(m, 0, rho[m, 0])) +  delta[prev_kp, m-1]
                    else: ## k1 == kp1 & k2 != kp2
                        tempVal[ prev_kp] = log(probJ(m, 1, rho[m, 0])) + log( alpha[ m, k2]) + delta[ prev_kp, m-1]
                elif ( k2 == kp2): ## k1 != kp1 & k2 == kp2
                    tempVal[ prev_kp] = log(probJ(m, 1, rho[m, 0])) + log( alpha[ m, k1]) +  delta[ prev_kp, m-1]
                elif ( k1 == kp2) or (k2 == kp1): ## crossing
                    tempVal[ prev_kp] = log(probJ(m, 1, rho[m, 0])) + log( alpha[ m, k1]) + delta[ prev_kp, m-1]
                else: ## k1 !=kp1 & k2 != kp2
                    tempVal[ prev_kp] = log(probJ(m, 2, rho[m, 0])) + log( alpha[ m, k1]) + alpha[m, k2] +  delta[ prev_kp, m-1]
            psi[ ikp, m] = argmax(tempVal, npairs)
            delta[ ikp, m] = tempVal[ psi[ ikp, m]] + log( genprG( theta[ m, k1], theta[ m, k2], gen[m]))

    ## termination
    soluce[ nLoc - 1 ] = argmax( delta[ :, nLoc-1], npairs)
    for m in range( nLoc - 2, -1, -1):
        soluce[ m ] = psi[ soluce[ m+1 ], m+1]
    return [ idx2pair(ikp, nK) for ikp in soluce]
            
cpdef genCalc(aa,tt,rr,gg,u2p):
    cdef np.ndarray[np.float64_t, ndim=2] alpha=aa
    cdef np.ndarray[np.float64_t,ndim=2] theta=tt
    cdef np.ndarray[np.float64_t, ndim=2] rho=rr
    cdef np.ndarray[np.int_t, ndim=1] gen=gg
    cdef int up2pz=u2p
    ## implementation comment: TODO = try to minimize the memory footprint
    ## of big Nloc*nK*nK matrices

    ## cython declarations
    cdef int tScale
    cdef double dummy,t1,t2
    cdef double temp,tScaleTemp
    cdef double normC
    cdef intnLoc,nK
    cdef int k,k1,k2,m
    cdef double logLikelihood
    ## end cython declarations

    nLoc=alpha.shape[0]
    nK=alpha.shape[1]
    ##
    ## compute backward probabilities
    ##
    cdef np.ndarray[np.int_t,ndim=1] betaScale=np.zeros(nLoc,dtype=np.int)
    cdef np.ndarray[np.float64_t,ndim=2] tSumk=np.zeros((nLoc,nK),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] tDoubleSum=np.zeros(nLoc,dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=3] mBeta=np.zeros((nLoc,nK,nK),dtype=np.float64)

    for k1 in range(nK):
        for k2 in range(nK):
            mBeta[nLoc-1,k1,k2]=1
    ## calc the marginal sum at locus M as in appendix A
    for k1 in range(nK):
        for k2 in range(nK):
            tSumk[nLoc-1,k1] += genprG(theta[nLoc-1,k1],theta[nLoc-1,k2],gen[nLoc-1])*mBeta[nLoc-1,k1,k2]*alpha[nLoc-1,k2]
        tDoubleSum[nLoc-1]+=tSumk[nLoc-1,k1]*alpha[nLoc-1,k1]
    ## recurrent calculations backward
    for m in range(nLoc-1,0,-1):
        ## these loops could be parallelized across cluster pairs #CUDA
        for k1 in range(nK):
            for k2 in range(k1,nK):
                temp=0.5*probJ(m,1,rho[m,0])*(tSumk[m,k1]+tSumk[m,k2])
                temp+=probJ(m,0,rho[m,0])*genprG(theta[m,k1],theta[m,k2],gen[m])*mBeta[m,k1,k2]
                mBeta[m-1,k1,k2]=temp+probJ(m,2,rho[m,0])*tDoubleSum[m]
                mBeta[m-1,k2,k1]=mBeta[m-1,k1,k2]
        ## marginal sum and real sum for loc m-1
        for k1 in range(nK):
            for k2 in range(nK):
                tSumk[m-1,k1]+=genprG(theta[m-1,k1],theta[m-1,k2],gen[m-1])*mBeta[m-1,k1,k2]*alpha[m-1,k2]
            tDoubleSum[m-1]+=tSumk[m-1,k1]*alpha[m-1,k1]
        tScaleTemp=np.sum(mBeta[m-1])
        ## tScaleTemp=0 a gerer <---
        if tScaleTemp<1e-20:
            tScale=20
        else:
            tScale=int(-log(tScaleTemp)/log(10))
        if tScale <=0:
            tScale=0
        betaScale[m-1]=betaScale[m]+tScale
        if tScale>0:
            dummy=myPow10(tScale)
            tDoubleSum[m-1]*=dummy
            for k1 in range(nK):
                tSumk[m-1,k1]*=dummy
                mBeta[m-1,k1,k1]*=dummy
                for k2 in range(k1+1,nK):
                    mBeta[m-1,k1,k2]*=dummy
                    mBeta[m-1,k2,k1]*=dummy
    ##
    ## compute forward probabilities
    ##
    cdef np.ndarray[np.float64_t,ndim=3] mPhi=np.zeros((nLoc,nK,nK),dtype=np.float64)
    cdef np.ndarray[np.int_t,ndim=1] phiScale=np.zeros(nLoc,dtype=np.int)
    ## at locus 0
    for k1 in range(nK):
        for k2 in range(k1,nK):
            mPhi[0,k1,k2]=alpha[0,k1]*alpha[0,k2]*genprG(theta[0,k1],theta[0,k2],gen[0])
            mPhi[0,k2,k1]=mPhi[0,k1,k2]
    ## calc the marginal sum at locus 0 (appx A)
    tDoubleSum[0]=0
    for k1 in range(nK):
        tSumk[0,k1]=0
        for k2 in range(nK):
            tSumk[0,k1]+=mPhi[0,k1,k2]
        tDoubleSum[0]+=tSumk[0,k1]
    tScale=0
    if tDoubleSum[0] != 0:
        tScale=int(-log(tDoubleSum[0])/log(10))
        if tScale <0:
            tScale=0
        phiScale[0]=tScale
        if tScale>0:
            dummy=myPow10(tScale)
            for k1 in range(nK):
                tSumk[0,k1]*=dummy
                mPhi[0,k1,k1]*=dummy
                for k2 in range(k1+1,nK):
                    mPhi[0,k1,k2]*=dummy
                    mPhi[0,k2,k1]*=dummy
    # do the reccurence
    for m in range(nLoc-1):
        ## This loop can be parallelized across cluster pairs #CUDA
        for k1 in range(nK):
            for k2 in range(k1,nK):
                temp=alpha[m+1,k1]*tSumk[m,k2]+alpha[m+1,k2]*tSumk[m,k1]
                temp*=0.5*probJ(m+1,1,rho[m+1,0])
                temp+=probJ(m+1,0,rho[m+1,0])*mPhi[m,k1,k2]
                temp+=probJ(m+1,2,rho[m+1,0])*alpha[m+1,k1]*alpha[m+1,k2]*tDoubleSum[m]
                mPhi[m+1,k1,k2]=temp*genprG(theta[m+1,k1],theta[m+1,k2],gen[m+1])
                mPhi[m+1,k2,k1]=mPhi[m+1,k1,k2]
        tDoubleSum[m+1]=0
        for k1 in range(nK):
            tSumk[m+1,k1]=0
            for k2 in range(nK):
                tSumk[m+1,k1]+=mPhi[m+1,k1,k2]
            tDoubleSum[m+1]+=tSumk[m+1,k1]
        tScale=0
        if tDoubleSum[m+1]<=0:
            phiScale[m+1]=phiScale[m]
        else:
            tScale=int(-log(tDoubleSum[m+1])/log(10))
            if tScale<0:
                tScale=0
            phiScale[m+1]=phiScale[m]+tScale
            if tScale>0:
                dummy=myPow10(tScale)
                tDoubleSum[m+1]*=dummy
                for k1 in range(nK):
                    tSumk[m+1,k1]*=dummy
                    mPhi[m+1,k1,k1]*=dummy
                    for k2 in range(k1+1,nK):
                        mPhi[m+1,k1,k2]*=dummy
                        mPhi[m+1,k2,k1]*=dummy
    ## end mPhi
    logLikelihood=log(tDoubleSum[nLoc-1])/log(10)-phiScale[nLoc-1]
    ##
    ## compute Individual Contribution top,bottom,jmk
    ##
    # calc ProbZ
    cdef np.ndarray[np.float64_t,ndim=3] probZ=mPhi*mBeta
    cdef np.ndarray[np.float64_t,ndim=2] jmk=np.zeros((nLoc,nK),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2] top=np.zeros((nLoc,nK),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2] bot=np.zeros((nLoc,nK),dtype=np.float64)
    ## normalize
    for m in range(nLoc):
        normC=0
        for k1 in range(nK):
            normC+=probZ[m,k1,k1]
            for k2 in range(k1+1,nK):
                normC+=2*probZ[m,k1,k2]
        probZ[m]/=normC
    if up2pz>0:
        return probZ
    # calc jmk
    for k1 in range(nK):
        ##jmk[0,k1]=2*alpha[0,k1]
        jmk[0, k1] = probZ[0,k1,k1] 
        for k2 in range(nK):
            jmk[0, k1] += probZ[0, k2, k1] 
            
    for m in range(1,nLoc):
        dummy = myPow10(phiScale[m-1]+betaScale[m]-phiScale[nLoc-1])
        for k in range(nK):
            for k1 in range(nK):
                temp = tSumk[ m-1, k1] * probJ(m,1,rho[m,0])
                temp += 2 * probJ( m, 2, rho[m,0]) * tDoubleSum[m-1] * alpha[m,k1]
                jmk[m,k] += temp * genprG( theta[m,k], theta[m,k1], gen[m]) * mBeta[m,k,k1]
            jmk[m,k] *= alpha[m,k]
            jmk[m,k] /= tDoubleSum[nLoc-1]
            jmk[m,k] /= dummy
    # calc top,bottom
    for m in range(nLoc):
        ## bottom
        for k in range(nK):
            bot[m,k]+=probZ[m,k,k]
            for k1 in range(nK):
                bot[m,k]+=probZ[m,k1,k]
        ## top
        if gen[m]==0:
            for k in range(nK):
                top[m,k]=0
        elif gen[m]==1:
            for k in range(nK):
                for k1 in range(nK):
                    t1=theta[m,k]*(1-theta[m,k1])
                    t2=t1+theta[m,k1]*(1-theta[m,k])
                    if (k == k1):
                        top[m,k] += 2*probZ[m,k,k1]*t1/t2
                    else:
                        top[m,k] += probZ[m,k,k1]*t1/t2
        elif gen[m]==2:
            for k in range(nK):
                top[m,k] += probZ[m,k,k]
                for k1 in range(nK):
                    top[m,k] += probZ[m,k,k1]
        else:
            for k in range(nK):
                top[m,k]=0
                bot[m,k]=0
    return logLikelihood,top,bot,jmk

#### Haplotype Calculations
    
cdef double happrG(double t,int s):
    if s==0:
        return 1-t
    elif s==1:
        return t
    else:
        return 1
  
cpdef hapViterbi( aa, tt, rr, hh):
    cdef np.ndarray[np.float64_t, ndim=2] alpha = aa
    cdef np.ndarray[np.float64_t,ndim=2] theta = tt
    cdef np.ndarray[np.float64_t, ndim=2] rho = rr
    cdef np.ndarray[np.int_t, ndim=1] hap = hh

    ## cython declarations
    cdef int nLoc, nK
    cdef int k, prev_k, best_k, m
    cdef double best_v
    nLoc = alpha.shape[0]
    nK = alpha.shape[1]
    cdef np.ndarray[ np.float64_t, ndim = 2 ] delta = np.zeros((nK, nLoc), dtype=np.float64)
    cdef np.ndarray[ np.int_t, ndim = 2 ] psi = np.zeros((nK, nLoc), dtype=np.int)
    cdef np.ndarray[ np.int_t, ndim = 1] soluce = np.empty(nLoc, dtype= np.int)
    cdef np.ndarray[ np.float64_t, ndim = 1] tempVal = np.zeros( nK, dtype=np.float64)
    
    ## initialization
    for k in range(nK):
        delta[k,0] = log(alpha[ 0, k]) + log(happrG( theta[ 0, k], hap[0]))

    ## recursion
    for m in range(1, nLoc):
        for k in range(nK):
            for prev_k in range(nK):
                ## jump
                tempVal[ prev_k] = log(rho[ m, 0])+log( alpha[ m, k])
                if k == prev_k:
                    ## no jump
                    tempVal[ prev_k] = log( rho[ m, 0]*alpha[ m, k] + (1 - rho[ m, 0]))
                tempVal[ prev_k] += log( delta[prev_k, m-1])
                psi[ k, m] = argmax(tempVal, nK)
                delta[ k, m] = tempVal[ psi[ k, m]] +log( happrG( theta[ m, k], hap[ m]))
    
    ## termination
    soluce[ nLoc - 1 ] = argmax( delta[ :, nLoc-1], nK)
    for m in range( nLoc - 2, -1, -1):
        soluce[ m ] = psi[ soluce[ m+1 ], m+1]
    return soluce
            
cpdef hapCalc(aa,tt,rr,hh,u2p):
    cdef np.ndarray[np.float64_t, ndim=2] alpha = aa
    cdef np.ndarray[np.float64_t,ndim=2] theta = tt
    cdef np.ndarray[np.float64_t, ndim=2] rho = rr
    cdef np.ndarray[np.int_t, ndim=1] hap = hh
    cdef int up2pz=u2p
    ## cython declarations
    cdef int nLoc,nK,tScale
    cdef int k,m
    cdef double dummy,tScaleTemp,temp
    cdef double normC
    cdef double logLikelihood
    ## end cython declarations
    nLoc=alpha.shape[0]
    nK=alpha.shape[1]
    ##
    ## compute backward probabilities
    ##
    cdef np.ndarray[np.int_t,ndim=1] betaScale=np.zeros(nLoc,dtype=np.int)
    cdef np.ndarray[np.float64_t,ndim=1] tSum=np.zeros(nLoc,dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2] mBeta=np.zeros((nLoc,nK),dtype=np.float64)

    for k in range(nK):
        mBeta[nLoc-1,k]=1

    for k in range(nK):
        tSum[nLoc-1] += happrG(theta[nLoc-1,k],hap[nLoc-1])*alpha[nLoc-1,k]
    for m in range(nLoc-1,0,-1):
        tScaleTemp=0
        ## this loop can be parallelized across clusters #CUDA
        for k in range(nK):
            temp=(1.0-rho[m,0])*happrG(theta[m,k],hap[m])*mBeta[m,k]
            mBeta[m-1,k]=temp+rho[m,0]*tSum[m]
            tSum[m-1]+=happrG(theta[m-1,k],hap[m-1])*mBeta[m-1,k]*alpha[m-1,k]
            tScaleTemp+=mBeta[m-1,k]
        # if np.isnan(tScaleTemp):
        #     print mBeta[m-1]
        #     print mBeta[m]
        #     print alpha[m-1]
        #     print theta[m]
        #     print rho[m]
        #     print hap[m]
        #     print 'BLA'
        tScale=int(-log(tScaleTemp)/log(10))
        if tScale<0:
            tScale=0
        betaScale[m-1]=betaScale[m]+tScale
        if tScale >0:
            dummy=myPow10(tScale)
            tSum[m-1]*=dummy
            for k in range(nK):
                mBeta[m-1,k]*=dummy
    ##
    ## compute forward probabilities
    ##
    cdef np.ndarray[np.float64_t,ndim=2] mPhi=np.zeros((nLoc,nK),dtype=np.float64)
    cdef np.ndarray[np.int_t,ndim=1] phiScale=np.zeros(nLoc,dtype=np.int)
    for k in range(nK):
        mPhi[0,k]=alpha[0,k]*happrG(theta[0,k],hap[0])
    ## calc the marginal sum at locus 0 (appx A)
    tSum[0]=0
    for k in range(nK):
        tSum[0]+=mPhi[0,k]
    ## calc Phi 
    for m in range(nLoc-1):
        tSum[m+1]=0
        ## this loop could be parallelized across clusters #CUDA
        for k in range(nK):
            temp=(1-rho[m+1,0])*mPhi[m,k]+rho[m+1,0]*alpha[m+1,k]*tSum[m]
            mPhi[m+1,k]=temp*happrG(theta[m+1,k],hap[m+1])
            tSum[m+1]+=mPhi[m+1,k]
        tScale=0
        if tSum[m+1] < 0:
            phiScale[m+1]=phiScale[m]
        else:
            tScale=int(-log(tSum[m+1])/log(10))
        if tScale < 0:
            tScale=0
        phiScale[m+1]=phiScale[m]+tScale
        if tScale > 0:
            dummy=myPow10(tScale)
            tSum[m+1]*=dummy
            for k in range(nK):
                mPhi[m+1,k]*=dummy
    ## end calc Phi
    logLikelihood=log(tSum[nLoc-1])/log(10)-phiScale[nLoc-1]
    ##
    ## compute individual contributions top,bottom,jmk
    ##
    # compute probZ see appx A at the end
    cdef np.ndarray[np.float64_t,ndim=2] probZ=mPhi*mBeta
    cdef np.ndarray[np.float64_t,ndim=2] jmk=np.zeros((nLoc,nK),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2] top=np.zeros((nLoc,nK),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2] bot=np.zeros((nLoc,nK),dtype=np.float64)
    
    for m in range(nLoc):
        normC=0
        for k in range(nK):
            normC+=probZ[m,k]
        for k in range(nK):
            probZ[m,k]/=normC
    if up2pz:
        return probZ
    ## compute expected jum prob for each interval (appx C)
    # locus 0
    for k in range(nK):
        ##jmk[0,k]=alpha[0,k]
        jmk[0, k] = probZ[ 0, k]
    for m in range(1,nLoc):
        dummy=myPow10(phiScale[m-1]+betaScale[m]-phiScale[nLoc-1])
        for k in range(nK):
            jmk[m,k]  = tSum[m-1]*rho[m,0]*happrG(theta[m,k],hap[m])*mBeta[m,k]
            jmk[m,k] *= alpha[m,k]
            jmk[m,k] /= tSum[nLoc-1]
            jmk[m,k] /= dummy
            # if not np.isfinite(jmk[m,k]):
            #     print tSum[m-1],rho[m,0],theta[m,k],hap[m],mBeta[m,k]
            #     print alpha[m,k]
            #     print tSum[nLoc-1]
            #     print dummy,phiScale[m-1],betaScale[m],phiScale[nLoc-1]
            #     raise ValueError
    # calc thetablock and its inner product with probZ (appx C.)
    for m in range(nLoc):
        if hap[m] == 0:
            for k in range(nK):
                bot[m,k]=probZ[m,k]
        elif hap[m]==1:
            for k in range(nK):
                top[m,k]=probZ[m,k]
                bot[m,k]=probZ[m,k]
    return logLikelihood,top,bot,jmk


