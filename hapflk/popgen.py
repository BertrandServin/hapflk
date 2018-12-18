import sys
import os
import pickle
import tempfile
import numpy as np
from scipy.stats import chi2,norm
from hapflk import nj
import warnings

debug=False
if not debug:
    np.seterr(all='ignore')
    warnings.filterwarnings('ignore',category=FutureWarning)

def reynolds_multi_allelic(Mf,nloc,decompose=False):
    '''
    Compute Reynolds distances from multiallelic loci

    Parameters:
    -----------
    
    Mf : numpy array of allele frequencies in populations
         rows are populations
         columns are allele frequencies, by loci
    nloc : number of loci

    Returns:
    --------

    dist : numpy array (n x n) of reynolds genetic distances
    numerator    : numerator of the Reynolds Distance
    denominator  : denominator of the Reynolds Distance
    '''
    npop=Mf.shape[0]
    A=np.dot(Mf,Mf.T)
    dist=np.zeros((npop,npop))
    if decompose:
        numerator=np.zeros((npop,npop))
        denominator=np.zeros((npop,npop))
    else:
        numerator=None
        denominator=None
    for i in range(npop-1):
        for j in range(i+1,npop):
            dist[i,j]=0.5*(A[i,i]+A[j,j]-2*A[i,j])/(nloc-A[i,j])
            if decompose:
                numerator[i,j]=A[i,i]+A[j,j]-2*A[i,j]
                denominator[i,j]=2*(nloc-A[i,j])
    dist=dist+dist.T
    numerator=numerator+numerator.T
    denominator=denominator+denominator.T
    return dist,numerator,denominator
   
def reynolds(Mf):
    '''
    Compute Reynolds distances from SNP allele frequencies

    Parameters:
    ----------------

    Mf : numpy array of allele frequencies in populations
          rows are populations (n), columns are markers (p).

    Returns:
    -----------

    dist : numpy array (n x n) of reynolds genetic distances
    '''
    npop,nloc=Mf.shape
    dist=np.zeros((npop,npop))
    A=np.dot(Mf,Mf.T)+np.dot((1-Mf),(1-Mf).T)
    for i in range(npop-1):
        for j in range(i+1,npop):
            dist[i,j]=0.5*(A[i,i]+A[j,j]-2*A[i,j])/(nloc-A[i,j])
    dist=dist+dist.T
    return dist

def heterozygosity(Mf):
    '''
    Compute heterozygosity from SNP allele frequencies
    Parameters:
    ----------------

    Mf : numpy array of allele frequencies in populations
          rows are populations (n), columns are markers (p).

    Returns:
    -----------

    hzy : numpy array (n x 1) of mean heterozygosities
    '''
    npop,nloc=Mf.shape
    hzy_func = lambda x: 2*x*(1-x)
    return np.average(hzy_func(Mf),axis=1)

def popKinship_fromFile(file_name,popnames):
    '''
    Read population kinship matrix from file

    File format is :

    Pop1 a11 a12 ...
    Pop2 a21 a22 ....

    where PopN is the population name (must be in popnames)
    aij is the kinship coefficient between pop i and j

    Parameters:
    -----------
    file_name : name of the file to read kinship from
    popnames : name of the pop to find in the file (order is preserved)

    See Also:
    ---------
    popKinship for estimating the kinship matrix
    '''

    data=[x.split() for x in open(file_name).readlines()]
    ##popnames2=popnames[:]
    # if outgroup is not None:
    #     popnames2.remove(outgroup)
    ordre=[popnames.index(x[0]) for x in data]
    if len(ordre) != len(data):
        print( 'Not Enough populations in kinship file !')
        raise ValueError
    FF=np.zeros((len(popnames),len(popnames)),dtype=float)
    for i in range(len(popnames)):
        ix=ordre[i]
        for j in range(len(popnames)):
            jx=ordre[j]
            FF[ix,jx]=float(data[i][j+1])
    return FF

def popKinship_new(D,popnames,outgroup=None,fprefix=None,keep_outgroup=False,hzy=None,dump_tree=True):
    '''
    Compute kinship matrix between populations

    Parameters:
    ---------------

    D : Reynolds distances between populations
    popnames : names of populations
    outgroup : population to use as outgroup for rooting (if None, use hzy (should give good results) 
               or do midpoint rooting (not implemented))
    fprefix : if set, save the temporary files with names built from fprefix
    keep_outgroup : do not drop the population used for rooting the tree
    hzy : Optimize root finding on observed heterozygosities vector hzy

    Returns:
    -----------

    Fij : Kinship matrix between populations
    fprefix_{ }.txt : if fprefix was set

    See Also:
    ------------

    popgen.reynolds to compute reynolds distances between populations
    popgen.heterozygosity to compute heterozygosities of populations
    '''
    if outgroup is not None:
        try:
            assert outgroup in popnames
        except AssertionError:
            print( "Outgroup not found in file",outgroup)
            print( *popnames)
            sys.exit(1)
    if not fprefix:
        buf,rey_file_name=tempfile.mkstemp(prefix='reynolds')
        buf,fij_file_name=tempfile.mkstemp(prefix='fij',suffix='.txt')
        buf,tree_file_name=tempfile.mkstemp(prefix='tree',suffix='.txt')
        buf,treedump_file_name=tempfile.mkstemp(prefix='treedump',suffix='.tree')
    else:
        rey_file_name=fprefix+'_reynolds.txt'
        fij_file_name=fprefix+'_fij.txt'
        tree_file_name=fprefix+'_tree.txt'
        treedump_file_name=fprefix+'.tree'
    ## write reynolds to file
    reynolds_out=open(rey_file_name,'w')
    for i in range(D.shape[0]):
        tw=[popnames[i]]
        for j in range(D.shape[1]):
            tw.append(str(D[i,j]))
        print(' '.join(tw), file = reynolds_out)
    reynolds_out.close()
    ## Build the tree
    dist_mat=2*D
    my_nj=nj.NJ(dist_mat,popnames)
    my_nj.fit()
    if outgroup:
        root_edge=[e for e in my_nj.edges if e.n1.label==outgroup or e.n2.label==outgroup][0]
        tree=nj.Rooted_Tree()
        if keep_outgroup:
            tree.build_from_edges(my_nj.edges,root_edge)
            if hzy is not None:
                ## we can optimize root placement
                hzy_dict={}
                for i,nom in enumerate(popnames):
                    hzy_dict[nom]=hzy[i]
                tree.optim_root(hzy_dict)
        else:
            print( "Rooting with %s and dropping it."%outgroup)
            tree.build_from_edges(my_nj.edges,root_edge,outgroup=outgroup)
    else:
        ## No outgroup information
        if hzy is not None:
            hzy_dict={}
            for i,nom in enumerate(popnames):
                hzy_dict[nom]=hzy[i]
            ## we can optimize using heterozygosities
            resid=np.inf ## optim criterion
            root_edge_best=None ## where we will place the root
            for candidate in my_nj.edges:
                edges=[e for e in my_nj.edges]
                tree=nj.Rooted_Tree()
                tree.build_from_edges(edges,candidate)
                tree_fit=tree.optim_root(hzy_dict)
                ##print 'Edge',candidate,' : ',tree_fit.fun
                if tree_fit.x[1]>0 and tree_fit.fun<resid:
                    resid=tree_fit.fun
                    root_edge_best=candidate
                    root_pos_best=tree_fit.x[1]
            tree=nj.Rooted_Tree()
            tree.build_from_edges(my_nj.edges,root_edge_best)
            tree.shift_root(root_pos_best)
        else:
            ## no hzy information, should revert to midpoint but this is
            ## usually not a good idea ...
            raise NotImplementedError
    ## Now we have a tree, get Kinship
    with open(tree_file_name,'w') as f:
        print( tree.newick(), file = f)
    if dump_tree:
        with open(treedump_file_name,'w') as f:
            pickle.dump(tree,f)
    kin,poplabels=tree.kinship()
    with open(fij_file_name,'w') as f:
        for i,nom in enumerate(poplabels):
            print( nom,' '.join([str(x) for x in kin[i,]]), file = f)
    ### we have to be careful because the pop order can change within Fij
    if outgroup and not keep_outgroup:
        pname_temp=[x for x in popnames if x!=outgroup]
    else:
        pname_temp=popnames[:]
    ordre=[poplabels.index(x) for x in pname_temp]
    FF=kin[ordre,:][:,ordre]
    ## clean up temp files if needed
    if not fprefix:
        os.remove(rey_file_name)
        os.remove(fij_file_name)
        os.remove(tree_file_name)
        if dump_tree:
            os.remove(treedump_file_name)
    return FF

class FLK_result():
    def __init__(self):
        self.p0=np.nan
        self.val=0
        self.pval=1
        
class FLK_test():
    '''
    FLK test implementation
    '''
    def __init__(self,kinship,diallelic=True):
        '''
        Creates an FLK test

        Parameters:
        ---------------

        kinship : population kinship matrix
        diallelic : if True, the locus is diallelic (e.g. a SNP) o.w. multiallelic
        
        '''
        self.F=kinship
        assert self.F.shape[0]==self.F.shape[1]
        self.dimension=self.F.shape[0]
        self.diallelic=diallelic
        self.invF=np.linalg.inv(self.F)
        self.un = np.ones(self.dimension)
        self.w = np.dot(self.invF,self.un.T)/np.dot(self.un,np.dot(self.invF,self.un.T))
        self.D,self.Q = np.linalg.eigh(self.F)
        if diallelic:
            self.eigen_contrib=self.eigen_contrib_diallelic
        else:
            self.eigen_contrib=self.eigen_contrib_multi
        
    def eigen_contrib_diallelic(self,p,diallelic=True):
        '''
        Computes orthogonalized contributions of each component to the FLK
        statistic, using the spectral decomposition of the variance
        covariance matrix V(p) V=cste*F=cste*Q*D*Q.T where Q is the matrix
        of eigen vectors and D the corresponding diagonal matrix with
        eigen values on the diagonal.

        Parameters:
        ---------------

        p : vector of population allele frequencies at a locus

        Return Value:
        -----------------
        a tuple of :
            - p0hat : estimated allele frequency in the ancestral population
            - Z : contributions of each PC to the FLK statistic
            - cste : multiplying constant to the test
        the value of FLK is = C*sum(L^2)
        '''
        p0hat = np.dot(self.w.T,p)
        cste = ((1 - (1/(np.dot(self.un,np.dot(self.invF,self.un.T)))))/(p0hat*(1-p0hat)))
        Z=np.dot((p-(p0hat*self.un)),self.Q)
        Z=np.dot(Z,np.diag(np.sqrt(1/self.D)))
        Z *= np.sqrt(cste)
        # tmp=np.dot(p-(p0hat*self.un),self.invF*cste)
        # tmp=np.dot(tmp,(p-(p0hat*self.un).T))
        # print tmp,np.sum([z**2 for z in Z])
        # R = np.dot(self.Q.T,(p-(p0hat*self.un).T))
        # R = np.dot(np.diag(np.sqrt(1/self.D)),R)
        # R *= np.sqrt(cste)
        # print R.shape
        # Z=[R[i] for i in range(R.shape[0])]
        return p0hat,Z,cste

    def eigen_contrib_multi(self,p):
        '''
        Multiallelic version of eigen_contrib_diallelic
        
        Parameters:
        ---------------
        
        p : vector of population allele frequencies at a locus
        
        Return Value:
        -----------------
        a tuple of :
        - p0hat : estimated allele frequency in the ancestral population
        - Z : contributions of each PC to the FLK statistic
        - cste : multiplying constant to the test
        the value of FLK is = C*sum(L^2)
        
        See Also:
        ------------
        
        eigen_contrib_diallelic 
        '''
        p0hat = np.dot(self.w.T,p)
        Z=np.dot((p-(p0hat*self.un)),self.Q)
        Z=np.dot(Z,np.diag(np.sqrt(1/self.D)))
        return p0hat,Z,1.0
    
    def eval_flk(self,p):
        '''
        Computes the FLK statistic

        Parameters:
        ---------------

        p : vector of population allele frequencies

        Return Value:
        -----------------
        a list of :
           -- p0
           -- FLK value
           -- corresponding p-value
           -- each contribution of the PC in turn
        '''
        assert p.shape[0]==self.dimension
        p0,Z,C=self.eigen_contrib(p)
        val=sum([x**2 for x in Z])
        if self.diallelic and p0>0.05 and p0<0.95:
            pval=chi2.sf(val,self.dimension-1)
        else:
            pval=np.nan
        return [p0,val,pval]+Z.tolist()
      
