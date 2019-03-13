#!/usr/bin/env python3
import pickle
import sys
import argparse
import bz2
from hapflk import InputOutput as IO
from hapflk import popgen
import hapflk.fastphase as myfph
import numpy as np
from multiprocessing import cpu_count


class SNP(object):
    """ Pickable simplified SNP class"""
    def __init__(self, name, a1, a2):
        """
        SNP class to store name and alleles
        :type name: basestring
        :type a1: basestring
        :type a2: basestring
        """
        self.name = name
        self.alleles = [a1, a2]

class HapFLK(object):
    ''' A class to run an hapFLK analysis.
    '''
    def __init__(self):
        ''' Attributes of the class (not all need be specified)

        Note: Data arrays indices relate to the pops and sortes_snps lists.

            Input Data
            ----------
            - carte: data.Map instance of SNP info (may contain > nsnp)
            - sorted_snps: list of SNP objects ordered by position in carte
            - alleles: dict of SNP alleles (key=SNP names)
            - pops: list of populations

            Model Data
            -----------
            - outgroup: name of the outgroup used to build kinship. Usually not in pops
            - reynolds: matrix of reynolds distances between pop
            - hzy: heterozygosity of pops
            - kinship: Kinship matrix between pops
            - frq: SNP allele freq in pops
            - K : number of clusters
            - nfit : number of EM fits 
            - kfrq: haplotype cluster frequencies

            Test Data
            ---------
            - flk: FLK values
            - pzero: estimated ancestral allele frequencies
            - pval_FLK: theoretical p-value of FLK
            - hapflk: hapFLK values
            - eigvec: Eigen Vectors of kinship matrix
            - eigval: Eigen Values of kinship matrix
            - eigen_FLK: Spectral decomposition of FLK test
            - eigen_hapFLK: Spectral decomposition of hapFLK test
        '''
        ## SNP info
        self.carte=None
        self.sorted_snps=None
        self.alleles=None
        ## Model data
        self.pops=None
        self.outgroup=None
        self.reynolds=None
        self.hzy=None
        self.kinship=None
        self.frq=None
        self.K=-1
        self.nfit=0
        self.kfrq=None
        ## Test data
        self.flk=None
        self.pzero=None
        self.hapflk=None
        self.eigvec=None
        self.eigval=None
        self.flk_eigen=None
        self.hapflk_eigen=None

    @property
    def nsnp(self):
        return len(self.sorted_snps)

    @property
    def npop(self):
        return len(self.pops)
    
    @classmethod
    def from_db_file(cls,filename):
        '''Create Instance from dbfile'''
        with bz2.BZ2File(filename,'rb') as f:
            obj=pickle.load(f)
        obj.__class__=cls ## allows to load parent obj from subclasses
        return obj

    @classmethod
    def from_cmd_line(cls,opts):
        '''Create instance from command line arguments'''
        obj=cls()

        data=obj.read_input(opts)
        obj.model_setup(data,opts)
        obj.write_frq(opts.prefix)
        
        if opts.K>0:
            print("Fitting LD model (this might take a while)")
            fphpars=obj.run_fastphase(data,opts)
            print("Computing haplotype cluster frequencies")
            obj.calc_kfrq(data,opts,fphpars)
            if opts.kfrq:
                obj.write_kfrq(opts.prefix)
            obj.nfit,obj.K,np,ns=obj.kfrq.shape
        else:
            obj.kfrq=None
            obj.nfit=0
            obj.K=0
        return obj
        

    @staticmethod
    def populate_parser(parser):
        parser.add_argument('-p','--prefix',dest='prefix',help='prefix for output files',default='hapflk')
        parser.add_argument('--ncpu',metavar='N',help='Use N processors when possible',default=cpu_count(),type=int)
        parser.add_argument('--eigen',help='Perform eigen decomposition of tests',default=False,action='store_true')
        parser.add_argument('--reynolds',help='Force writing down Reynolds distances',default=False,action='store_true')
        parser.add_argument('--future',help=argparse.SUPPRESS,default=False,action="store_true") ## for testing future release
        parser.add_argument('--debug',help=argparse.SUPPRESS,default=False,action="store_true") ## for debug purpose
        parser.add_argument('--savedb',help=argparse.SUPPRESS,default=False,action="store_true") ## save hapflk DB
        parser.add_argument('--loaddb',dest='dbfile',help=argparse.SUPPRESS,default=None) ## load hapflk DB
        flk_opts=parser.add_argument_group('Population kinship ','Set parameters for getting the population kinship matrix')
        flk_opts.add_argument('--kinship',help='Read population kinship from file (if None, kinship is estimated)',metavar='FILE',default=None)
        flk_opts.add_argument('--reynolds-snps',dest='reysnps',type=int,help='Number of SNPs to use to estimate Reynolds distances',default=100000,metavar='L')
        flk_opts.add_argument('--outgroup',default=None,help='Use population POP as outgroup for tree rooting (if None, optimize root location)',metavar="POP")
        flk_opts.add_argument('--keep-outgroup',dest='keepOG',default=False,help='Keep outgroup in population set',action="store_true")
        flk_opts.add_argument('--covkin',default=False,help='Use covariance matrix as kinship', action='store_true')
        LD_opts=parser.add_argument_group('hapFLK and LD model','Switch on hapFLK calculations and set parameters of the LD model ')
        LD_opts.add_argument('-K',help='Set the number of clusters to K. hapFLK calculations switched off if K<0',default=-1,type=int)
        LD_opts.add_argument('--nfit',help='Set the number of model fit to use',type=int,default=10)
        LD_opts.add_argument('--phased','--inbred',help='Haplotype data provided',dest='inbred',action="store_true",default=False)
        LD_opts.add_argument('--fastLD',help=argparse.SUPPRESS,default=False,action='store_true') ## Fit LD model on "haploidized" individuals
        LD_opts.add_argument('--kfrq',dest='kfrq',help='Write Cluster frequencies (Big files)',action="store_true",default=False)
        LD_opts.add_argument('--write-params',dest='wparams',help=argparse.SUPPRESS,default=False,action='store_true')
        LD_opts.add_argument('--legacy', dest='legacy',help='Use Legacy fastPHASE', default=False, action='store_true')
        parser.add_argument('--annot',help='Shortcut for --eigen --reynolds --kfrq',default=False,action='store_true')
        parser.add_argument('--tree',help=argparse.SUPPRESS,default=False,action='store_true')

    @classmethod
    def get_opts(cls,args):
        ''' Parse arguments from a command line and returns hapFLK options'''
        ## Arguments and option parser
        parser=argparse.ArgumentParser(parents=[IO.io_parser],formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        cls.populate_parser(parser)
        if len(args)<2:
            parser.print_help()
            sys.exit(0)
        opts=parser.parse_args(args[1:])
        if opts.annot:
            opts.reynolds=True
            opts.kfrq=True
            opts.eigen=True
        if opts.sfile:
            opts.inbred=True
        return opts
    
    def read_input(self,opts):
        '''Read input files
        Creates the following attributes:
        -- carte
        -- sorted_snps
        -- nsnps
        '''
        myi=IO.parseInput(opts)
        if myi is None:
            self.parser.print_help()
            sys.exit(0)
        data=myi['dataset']
        self.carte=myi['map']
        self.sorted_snps=[]
        for sname in self.carte.sort_loci(data.snp.keys()):
            self.sorted_snps.append(SNP(sname,data.snp[sname].alleles[0],data.snp[sname].alleles[1]))
            
        return data
    
    def model_setup(self,data,opts):
        ''' Sets up FLK model. Creates the following attributes:
        -- reynolds 
        -- hzy
        -- outgroup
        -- kinship
        -- frq
        -- pops

        '''
        ## Compute allele frequencies
        print('Computing Allele Frequencies')
        res=data.compute_pop_frq()
        freqs=res['freqs']
        pops=res['pops']
        
        ## compute Reynolds distances
        print('Computing Reynolds distances')
        if opts.reysnps>self.nsnp:
            opts.reysnps=self.nsnp
        pbinom=float(opts.reysnps)/self.nsnp
        snp_subset=np.array(np.random.binomial(1,pbinom,self.nsnp),dtype=bool)
        self.reynolds=popgen.reynolds(freqs[:,snp_subset])
        self.hzy=popgen.heterozygosity(freqs[:,snp_subset])
        self.outgroup=opts.outgroup
        ## Get kinship
        if opts.kinship:
            print('Reading Kinship Matrix')
            if not opts.keepOG:
                self.kinship=popgen.popKinship_fromFile(opts.kinship,[pop for pop in pops if pop != self.outgroup])
            else:
                self.kinship=popgen.popKinship_fromFile(opts.kinship,[pop for pop in pops])
        else:
        ## estimate kinship
            print("Computing Kinship Matrix")
            self.kinship=popgen.popKinship_new(self.reynolds,pops,self.outgroup,
                                              fprefix=opts.prefix,
                                              keep_outgroup=opts.keepOG,
                                              hzy=self.hzy,
                                              dump_tree=opts.tree)
        ## store eigen decomp of kinship matrix
        self.eigval,self.eigvec=np.linalg.eigh(self.kinship)
        ## forget about the outgroup if we keep it
        if opts.keepOG:
            ## override results / initial value
            self.outgroup='Iwantmyoutgroupback'
        ## boolean array indicating kept populations
        keep_pop=np.array([x!=self.outgroup for x in pops],dtype=bool)
        self.frq=freqs[keep_pop,]
        self.pops=tuple([x for x in pops if x!=self.outgroup])
        if opts.covkin:
            print('Computing Covariance Matrix')
            ff=popgen.FLK_test(self.kinship)
            ## calculate hat(p0)
            pzero=np.array([ff.eigen_contrib(self.frq[:,s])[0] for s in range(self.nsnp)])
            sub=(pzero>0.05)&(pzero<0.95)
            ## calculate empirical covariance matrix
            myf=self.frq[:,sub]
            m=pzero[sub]
            s=np.sqrt(m*(1-m))
            myfstd=(myf-m)/s
            covkin=np.cov(myfstd,rowvar=True)
            ## Add a small term on the diagonal to ensure stable inverse
            val=np.trace(covkin)/self.npop
            assert (val >0)
            covkin=covkin+np.diag(np.ones(self.npop)*0.01*val)
            with open(opts.prefix+'_cov.txt','w') as fcov:
                for i,pi in enumerate(self.pops):
                    tw=[pi]
                    for j,pj in enumerate(self.pops):
                        tw.append(str(covkin[i,j]))
                    print(' '.join(tw),file=fcov)
            self.kinship=covkin
        
    def run_fastphase(self,data,opts):
        '''Estimates Scheet and Stephens LD model on data'''
        fphpars=[]
        with myfph.fastphase(self.nsnp, opts.ncpu, prfx = opts.prefix+'.fphlog') as fastphase_model:
            tohap=np.vectorize(lambda x: (x==1) and -1 or (x//2))
            sorted_snps_idx=np.array([data.snpIdx[s.name] for s in self.sorted_snps])
            for name,i in data.indivIdx.items():
                if data.indiv[name].pop==self.outgroup:
                    continue
                if opts.inbred:
                    haplo=np.array(tohap(np.array(data.Data[i,sorted_snps_idx],dtype=int)),dtype=np.int)
                    fastphase_model.addHaplotype(name,haplo)
                else:
                    fastphase_model.addGenotype(name,np.array(data.Data[i,sorted_snps_idx],dtype=int))
            if opts.legacy:
                for e in range(opts.nfit):
                    sys.stderr.write('\tEM %d / %d \r'%(e+1,opts.nfit))
                    sys.stderr.flush()
                    if opts.debug:
                        print()
                    par=fastphase_model.fit(nClus=opts.K,verbose=opts.debug, fast=opts.fastLD)
                    fphpars.append(par)
                    if opts.wparams:
                        fout=open(opts.prefix+'fph_par_'+str(e),'w')
                        par.write(fout)
                        fout.close()
            else:
                par = fastphase_model.optimfit(nClus=opts.K,verbose=opts.debug,nEM=opts.nfit, fast=opts.fastLD)
                # with open(opts.prefix+'debug_alpha_const.txt','w') as fpar:
                #     par.write(stream=fpar)
                ## update alpha
                # print('Finalizing: Updating Cluster Weights')
                # par.alpha_up = True
                # par = fastphase_model.fit(params=par,nClus=opts.K,verbose=opts.debug,nstep=50)
                # with open(opts.prefix+'debug_alpha_free.txt','w') as fpar:
                #     par.write(stream=fpar)
                opts.nfit = 1
                fphpars.append(par)
                if opts.wparams:
                        fout=open(opts.prefix+'fph_par','w')
                        par.write(fout)
                        fout.close()
            print()
        return fphpars
        
    def calc_kfrq(self,data,opts,fphpars):
        ''' Compute haplotype cluster population frequencies '''
        ## [E][K][npop x nsnp] array
        pop_cluster_freq=np.zeros((opts.nfit,opts.K,self.npop,self.nsnp),dtype=float)
        with myfph.fastphase( self.nsnp, opts.ncpu) as fastphase_model:
            tohap=np.vectorize(lambda x: (x==1) and -1 or (x/2))
            sorted_snps_idx=np.array([data.snpIdx[s.name] for s in self.sorted_snps])
            for ipop,popname in enumerate(self.pops):
                sys.stdout.write("\t %16s\r"%popname)
                sys.stdout.flush()
                pvec=data.populations[popname]
                n_pop_indiv=sum(pvec)
                for name,iind in data.indivIdx.items():
                    if pvec[iind]:
                        if opts.inbred:
                            haplo=np.array(tohap(np.array(data.Data[iind,sorted_snps_idx],dtype=int)),dtype=np.int)
                            fastphase_model.addHaplotype(name,haplo)
                        else:
                            fastphase_model.addGenotype(name,np.array(data.Data[iind,sorted_snps_idx],dtype=int))
                for ifit in range(opts.nfit):
                    imputations=fastphase_model.impute([fphpars[ifit]])
                    for nom,dat in imputations.items():
                        probZ=dat[1]
                        if opts.inbred:
                            pop_cluster_freq[ifit,:,ipop,:]+=np.transpose(probZ[0]/n_pop_indiv)
                        else:
                            pop_cluster_freq[ifit,:,ipop,:]+=np.transpose((0.5/n_pop_indiv)*(np.sum(probZ[0],axis=1)+np.sum(probZ[0],axis=2)))
                fastphase_model.flush()
        self.kfrq=pop_cluster_freq

    def run_tests(self,opts):
        if self.flk is None:
            print('Computing FLK tests')
            self.calc_flk()
            self.write_flk(opts.prefix)
            
        if self.K>0:
            if self.hapflk is None:
                print("Computing hapFLK tests")
                self.calc_hapflk()
                self.write_hapflk(opts.prefix)
        return
    def calc_flk(self):
        '''Computes FLK tests'''
        ## Compute single SNP FLK tests
        myFLK=popgen.FLK_test(self.kinship)
        myFLK_res=np.apply_along_axis(myFLK.eval_flk,0,self.frq)
        self.pzero=myFLK_res[0,]
        self.flk=myFLK_res[1,]
        self.pval_FLK=myFLK_res[2,]
        self.eigen_FLK=np.power(myFLK_res[3:,],2)
        return
    
    def calc_hapflk(self):
        ''' Compute hapFLK tests '''
        myFLK=popgen.FLK_test(self.kinship,diallelic=False)
        self.hapflk=np.zeros(self.nsnp,dtype=float)
        self.eigen_hapFLK=np.zeros((self.npop,self.nsnp),dtype=float)
        for e in range(self.nfit):
            for k in range(self.K):
                myFLK_res=np.apply_along_axis(myFLK.eval_flk,0,self.kfrq[e,k,])
                self.hapflk+=myFLK_res[1,]
                self.eigen_hapFLK+=np.power(myFLK_res[3:,],2)
        self.hapflk/=self.nfit
        self.eigen_hapFLK/=self.nfit
        return

    def calc_hapflk_ng(self):
        pass
    
    def savedb(self,prefix):
        ''' Save (pickle) instance of the class in a 'prefix'.db file
            NB: The instance can be loaded (unpickled) using the class method from_db_file.
        '''
        dbfile=bz2.BZ2File(prefix+'.db','wb')
        pickle.dump(self,dbfile,protocol=2)
        dbfile.close()

    def write_frq(self,prefix):
        ''' Write allele frequencies to 'prefix'.frq'''
        assert(self.frq is not None)
        with open(prefix+'.frq','w') as fout:
            print('rs','chr','pos','all_ref','all_alt',' '.join(self.pops),file=fout)
            for sidx,s in enumerate(self.sorted_snps):
                spos=self.carte.position(s.name)
                tw=[s.name,spos[0],int(spos[2]),s.alleles[1],s.alleles[0]]
                for ip,nom in enumerate(self.pops):
                    tw.append(self.frq[ip,sidx])
                print(*tw,file=fout)

    def write_kfrq(self,prefix):
        if self.kfrq is None:
            pass
        nfit,nclus,npop,nsnp = self.kfrq.shape
        for ifit in range(nfit):
            with bz2.open(prefix+'.fit_'+str(ifit)+'.bz2','wt') as fout:
                print('pop','locus','position','cluster','prob',file=fout)
                for ipop in range(npop):
                    for i,s in enumerate(self.sorted_snps):
                        spos=self.carte.position(s.name)
                        for ik in range(nclus):
                            print(self.pops[ipop],s.name,int(spos[2]),ik,self.kfrq[ifit,ik,ipop,i], file=fout)
            
    def write_flk(self,prefix):
        ''' Write FLK results to 'prefix'.flk'''
        assert(self.flk is not None)
        with open(prefix+'.flk','w') as fout:
            print('rs','chr','pos','pzero','flk','pvalue',file=fout)
            for sidx,s in enumerate(self.sorted_snps):
                spos=self.carte.position(s.name)
                tw=[s.name,spos[0],int(spos[2]),self.pzero[sidx],self.flk[sidx],self.pval_FLK[sidx]]
                print(*tw,file=fout)

    def write_hapflk(self,prefix):
        ''' Write hapFLK results to 'prefix'.hapflk'''
        assert(self.flk is not None)
        with open(prefix+'.hapflk','w') as fout:
            print('rs','chr','pos','hapflk','K',file=fout)
            for sidx,s in enumerate(self.sorted_snps):
                spos=self.carte.position(s.name)
                tw=[s.name,spos[0],int(spos[2]),self.hapflk[sidx],self.kfrq.shape[1]]
                print(*tw,file=fout)

    
 
