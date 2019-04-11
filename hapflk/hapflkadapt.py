from __future__ import print_function, division
import sys
import numpy as np
import scipy.stats as ss
from scipy.stats import chi2,norm, multivariate_normal
from scipy.stats import percentileofscore
from scipy.optimize import minimize as optim
from scipy import interpolate
from numpy.linalg import multi_dot
from multiprocessing import Pool

#### Base functions for parallel computations (to put into own file)
def pscore(args):
    a, val = args
    return percentileofscore(a,val,kind='weak')

def loglik_h0_qfunc(pzero,p,Vinv):
    ll = - p.shape[0] * np.log( pzero * (1-pzero))
    res = p - pzero
    ll -= multi_dot( (res.T, Vinv, res) ) / (pzero * (1-pzero))
    return -ll

def root_pzero_h0(args):
    p, pini, Vinv = args
    res = optim(loglik_h0_qfunc, pini, args=( p, Vinv), method='SLSQP', bounds = [ (0,1) ],options = {'maxiter':20})
    if not res.success:
        ## we do it by hand
        xx = np.linspace(np.min(p),np.max(p),100) ## pzero should be between min(p) and max(p)
        ll = [ loglik_h0_qfunc( pz, p, Vinv) for pz in xx]
        ill = np.argmin(ll)
        if ill == 0:
            xx = np.linspace(0,np.min(p),100)
            ll =  [ loglik_h0_qfunc( pz, p, Vinv) for pz in xx]
            ill = np.argmin(ll)
        elif ill == 99:
            xx = np.linspace(np.max(p),1,100)
            ll =  [ loglik_h0_qfunc( pz, p, Vinv) for pz in xx]
            ill = np.argmin(ll)
        res = xx[ ill ]
        return (res, 0)
    else:
        return (res.x[0], 1)
                    
def loglik_h1_qfunc(pzero,p,xVinv,jacobian,x,Vinv):
    '''Optimization function to minimize -logLikelihood (i.e. maximize logLikelihood) '''
    #num = np.dot( xVinv, p-pzero)
    beta = multi_dot( (jacobian, xVinv, p-pzero))
    ll = - p.shape[0] * np.log( pzero * (1-pzero))
    res = ( p - pzero - np.dot(x, beta) )
    ll -= multi_dot( (res.T, Vinv, res) ) / (pzero * (1-pzero))
    return -ll

def root_pzero(args):
    p, pini, xVinv, denum, x, Vinv = args
    res = optim(loglik_h1_qfunc,pini, args=(p,xVinv,denum,x,Vinv),method='SLSQP', bounds = [ (0,1) ],options = {'maxiter':20})
    if not res.success:
        ## we do it by hand
        xx = np.linspace(np.min(p),np.max(p),100) ## pzero should be between min(p) and max(p)
        ll = [ loglik_h1_qfunc(pz, p, xVinv,denum,x,Vinv) for pz in xx]
        ill = np.argmin(ll)
        if ill == 0:
            xx = np.linspace(1e-4,np.min(p),100)
            ll =  [ loglik_h1_qfunc( pz, p,xVinv,denum,x, Vinv) for pz in xx]
            ill = np.argmin(ll)
        elif ill == 99:
            xx = np.linspace(np.max(p),0.9999,100)
            ll =  [ loglik_h1_qfunc( pz, p,xVinv,denum,x, Vinv) for pz in xx]
            ill = np.argmin(ll)
        res = xx[ ill ]
        return (res, 0)
    else:
        return (res.x[0], 1)
    
def beta_h1(pzero,p,xVinv, denum):
    num = np.dot( xVinv, (p-pzero) )
    beta = np.dot( denum, num)
    return beta

def calc_beta_h1(args):
    pzero, p, xVinv, denum = args
    return beta_h1(pzero,p,xVinv, denum)

def loglik_h1( pzero, beta, frq, x, Vinv):
    npop = x.shape[0]
    ll   = - npop * np.log( pzero * (1-pzero))
    res  = ( frq - pzero - np.dot(x, beta) )
    ll  -= multi_dot( ( res.T, Vinv, res)) / (pzero * (1-pzero))
    return ll

def calc_loglik_h1(args):
    pzero, beta, p, x, Vinv = args
    return loglik_h1(pzero,beta,p,x,Vinv)

def loglik_h0(p, w, Vinv):
    npop=p.shape[0]
    pzero_hat = np.dot( w.T, p)
    ##pzero_hat = self.xx[ np.argmin( np.abs( pzero_hat - self.xx) ) ]
    ll = - npop * np.log( pzero_hat * (1 - pzero_hat))
    res = (p - pzero_hat)
    ll -= np.dot( res.T, np.dot( Vinv, res ) ) / (pzero_hat * (1 - pzero_hat) )
    return ll

def calc_loglik_h0(args):
    p, w, Vinv = args
    return loglik_h0(p, w, Vinv)

def calc_lrt(args):
    p, xVinv, denum, x, Vinv, w = args
    #### H1 Calculations
    # pini = np.mean(p)
    # res = optim(loglik_h1_qfunc, pini, args=(p, xVinv, denum, x, Vinv), method='SLSQP', bounds = [ (0,1) ],options = {'maxiter':20})
    # if not res.success:
    #     ## we do it by hand
    #     xx = np.linspace(np.min(p),np.max(p),100) ## pzero has to be between min(p) and max(p)
    #     ll = [ loglik_h1_qfunc(pz, p, xVinv,denum,x,Vinv) for pz in xx]
    #     pzero = xx[ np.argmin(ll) ]
    # else:
    #     pzero = res.x[0]
    ## beta
    pbar = np.mean(p)
    if (pbar < 1e-3) or (pbar > 0.999):
        pzero = pbar
        succ1 = 1
        beta = np.zeros(x.shape[1])
        ll1 = 0
        pzero_null=pbar
        ll0 = 0
        succ0 = 1
    else:
        pzero, succ1 = root_pzero( [ p, pbar, xVinv, denum, x, Vinv ])
        beta = multi_dot( (denum, xVinv, p-pzero))
        ll1 = -loglik_h1_qfunc( pzero, p, xVinv, denum, x, Vinv)
        #### H0 Calculations
        pzero_null, succ0 = root_pzero_h0( (p, pbar, Vinv))
        ll0 = -loglik_h0_qfunc( pzero_null, p, Vinv)
    succ = ( ( succ0 == 1 ) and (succ1 ==1) and 1) or 0
    lrt = ll1 - ll0
    if lrt <0 :
        lrt = 0
    res_h0 = { 'llik': ll0, 'pzero': pzero_null, 'pzsucc': succ0}
    res_h1 = { 'llik': ll1, 'pzero': pzero, 'beta': beta, 'pzsucc': succ1}
    return { 'H0': res_h0, 'H1': res_h1, 'lrt':lrt, 'pvalue': chi2.sf( lrt, x.shape[1]),'pzsucc':succ }

def loglik_h0_ML(p, w, Vinv):
    npop=p.shape[0]
    pzero_hat,_ = root_pzero_h0( (p, np.mean(p), Vinv ))
    ll = - npop * np.log( pzero_hat * (1 - pzero_hat))
    res = (p - pzero_hat)
    ll -= multi_dot( ( res.T, Vinv, res)) / (pzero_hat * (1 - pzero_hat) )
    return ll

def calc_lrt_ML_old(args):
    p, xVinv, denum, x, Vinv, w = args
    ## root_pzero
    pini = np.mean(p)
    res = optim(loglik_h1_qfunc, pini, args=(p, xVinv, denum, x, Vinv), method='SLSQP', bounds = [ (0,1) ],options = {'maxiter':20})
    if not res.success:
        ## we do it by hand
        xx = np.linspace(np.min(p),np.max(p),100) ## pzero has to be between min(p) and max(p)
        ll = [ loglik_h1_qfunc(pz, p, xVinv,denum,x,Vinv) for pz in xx]
        pzero = xx[ np.argmin(ll) ]
    else:
        pzero = res.x[0]
    ## beta
    beta = multi_dot( (denum, xVinv, p-pzero))
    ## LRT 
    ll1 = loglik_h1( pzero, beta, p, x, Vinv)
    ll0 = loglik_h0_ML( p, w, Vinv)
    return { 'pzero': np.dot(w.T, p), 'lrt':ll1 - ll0}


def bayes_factor_vect(p, xx, logdet_omega, logdet_nu, logdet_V, beta_mean_fact, omega_inv, Vinv):
    npop = p.shape[0]
    if len(p.shape) == 1:
        frq = p.reshape( npop, 1)
    else: 
        frq = p
    ns = frq.shape[1]
    pzero = np.broadcast_to( xx, ( ns, len(xx)) )

    ## scaling factor
    lp = - npop * np.log( pzero * (1-pzero))
    lp_1 = lp + logdet_omega - logdet_nu - logdet_V
    lp_0 = lp - logdet_V
    ## SNP Sum of Squares
    res = (frq - pzero).T
    res=res.reshape( len(xx), npop, ns)
    postmean = np.einsum( 'ij,kjl->kil', beta_mean_fact, res)
    betass = np.einsum( 'ij,kjl->kil', omega_inv, postmean)
    BT = np.einsum('ijk->ikj',postmean)
    betass = np.einsum( 'ijk,ikj->i',BT,betass)
    datass=np.einsum('ij,kjl->kil',Vinv,res)
    RT=np.einsum('ijk->ikj',res)
    datass=np.einsum('ijk,ikj->i',RT,datass)
    ## compute Bayes Factor
    lp_0 -= datass / (pzero * (1-pzero))
    lp_1 -= ( datass - betass) / ( pzero * ( 1 - pzero))
    vmax0 = np.max(lp_0)
    P0 = np.log( np.sum( np.exp( lp_0 - vmax0) ) ) + vmax0
    vmax1 = np.max(lp_1)
    P1 = np.log( np.sum( np.exp( lp_1 - vmax1) ) ) + vmax1
    return (P1-P0)/np.log(10)
    
def calc_bf(args):
    p, xx, logdet_omega, logdet_nu, logdet_V, beta_mean_fact, omega_inv, Vinv = args
    pbar = np.mean(p)
    if (pbar < 1e-2) or (pbar > 0.99):
        return 0
    return bayes_factor_vect(p, xx, logdet_omega, logdet_nu, logdet_V, beta_mean_fact, omega_inv, Vinv)

def average_log10bf(logbf):
    '''Computes the average of a list of log10(bf)'''
    vmax = np.max(logbf)
    bfm = np.log10( np.mean( np.power(10, logbf - vmax))) + vmax 
    return bfm

def stdize_covar(x):
    try:
        ncovar = x.shape[1]
    except IndexError:
        x = x.reshape( x.shape[0], 1)
        ncovar = 1
    assert len(x.shape) == 2
    sx = np.zeros(x.shape)
    for ic in range(ncovar):
        m = np.mean( x[ :, ic])
        s = np.std( x[ :, ic], ddof=1)
        assert s>0
        sx[:, ic] = (x[ :, ic] - m)/s
    return sx
        
###### MAIN Class
class FLKadapt(object):
    def __init__( self, kinship, x, prefix = 'hapflkadapt', pz_step=0.01, ncpu=4, stdize=True):
        self.prefix=prefix
        self.standardized=False
        if stdize:
            self.standardized = True
            self.x = ss.zscore(x, axis=0, ddof=1) #stdize_covar(x)
        else:
            self.x = x - np.mean(x, axis=0)
            ##print("Covariables not standardized, proceed with care ... ")
        self.F = kinship
        self.diagFbar = np.mean( np.diag( self.F)) ## average distance to root
        _,self.logdet_V = np.linalg.slogdet( self.F)
        self.U,self.S,_ = np.linalg.svd(self.F)
        self.df = self.x.shape[1]
        self.Vinv = np.linalg.inv(self.F) 
        self.xVinv = np.dot(self.x.T,self.Vinv) # x'Vinv
        self.xVinvx = multi_dot( ( self.x.T, self.Vinv, self.x) )
        self.denum = np.linalg.inv( self.xVinvx ) # ( x' Vinv x )^{-1}
        self.npop=self.F.shape[0]
        self.un = np.ones(self.npop)
        self.w = np.dot(self.Vinv,self.un.T)/multi_dot( ( self.un, self.Vinv, self.un.T))
        self.pz_step=pz_step
        self.xx = np.arange(pz_step,1,step=pz_step)
        self.nproc=ncpu
        self.logfile=sys.stdout
        self.pool=None
        ## NULL calibrations
        self.bf_null = None
        self.correct_lrt = None
        
    def __enter__(self):
        self.pool = Pool(self.nproc)
        self.logfile = open(self.prefix+'.log','w')
        return self

    def __exit__(self,*args):
        self.pool.terminate()
        self.logfile.close()
        
    ## LRT Single SNP
    def calibrate_null( self, nsim = 10000, pmin=0.02):
        ''' 
        Create predictor of LRT correcting for inflation on low pzero values.

        1. Simulate frequency data (p) from MVN (pzero, self.F * (p0 * (1-p0))) for 
           p0 in [pmin, 1-pmin], on a log10 scale
        2. Compute LRT(p,x|p0)
        3. Adjust a spline on (x, LRT(p,x|p0)) and create a predictor f(LRT(p,x)) 
           that estimates #df of Chisq | p0
        '''
        assert self.pool is not None
        xp = 10**(np.linspace(np.log10(pmin),np.log10(0.5),20,endpoint=False))
        lrt_res = []
        bf_res = []
        args=[]
        for pzero in xp:
            pm = pzero * np.ones( self.npop)
            mvnorm = multivariate_normal( mean=pm, cov=self.F*pzero*(1-pzero))
            frq = mvnorm.rvs(nsim)
            frq[ frq < 0] = 0
            frq[ frq > 1] = 1
            for isim in range(nsim):
                args.append((frq[isim,], self.xVinv, self.denum, self.x, self.Vinv, self.w))
            # lrt_res += list( np.apply_along_axis( self.fit_loglik_optim, 1, frq, skip_small = False, correct = False))
            bf_res += list(  np.apply_along_axis( self.bayes_factor, 1, frq))
        lrt_res += self.pool.map(calc_lrt, args)
        x = []
        y = []
        for v in lrt_res:
            if not np.isnan(v['lrt']):
              x.append( v['H1']['pzero'])
              x.append( 1-v['H1']['pzero'])
              y.append( v['lrt'])
              y.append( v['lrt'])
        x = np.array(x)
        y = np.array(y)
        idx = np.argsort(x)
        knots = sorted( np.concatenate( [ xp, 1-xp]))[1:-1]
        self.correct_lrt = interpolate.LSQUnivariateSpline(x[idx], y[idx],knots)
        self.bf_null = sorted(bf_res)
        with open(self.prefix+'.calib','w') as f:
            print('# LRT DF',file=f)
            for xx in knots:
                print(xx,self.correct_lrt(xx),file=f)
            print('# BF percentiles',file=f)
            ntot=len(xp)*nsim
            probs=np.array([1.0/ntot, 1.0/nsim, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975, 0.99, 0.995, 1-1.0/nsim, 1-1.0/ntot])
            pc = np.percentile(bf_res, q=100*probs)
            for i,q in enumerate(probs):
                print(q,pc[i],file=f)
        
    def beta_h1(self,pzero,p):
        num = np.dot( self.xVinv, (p-pzero) )
        beta = np.dot(self.denum, num)
        return beta
    
    def loglik_h1(self,pzero,p):
        beta = self.beta_h1(pzero,p)
        ll = - self.npop * np.log( pzero * (1-pzero))
        res = ( p - pzero - np.dot(self.x, beta) )
        ll -= np.dot( res.T, np.dot( self.Vinv, res) ) / (pzero * (1-pzero))
        return {'llik': ll, 'beta': beta}
    
    def loglik_h0(self,p):
        pzero_hat = np.dot( self.w.T, p)
        ##pzero_hat = self.xx[ np.argmin( np.abs( pzero_hat - self.xx) ) ]
        ll = - self.npop * np.log( pzero_hat * (1 - pzero_hat))
        res = (p - pzero_hat)
        ll -= np.dot( res.T, np.dot( self.Vinv, res ) ) / (pzero_hat * (1 - pzero_hat) )
        return {'llik': ll, 'pzero': pzero_hat}

    def fit_loglik(self,p):
        res_h0 = self.loglik_h0(p)
        profile_h1 = [ self.loglik_h1(pz,p) for pz in self.xx ]
        best_pz = np.argmax( [ x['llik'] for x in profile_h1 ] )
        res_h1 = profile_h1[ best_pz ]
        pzero_hat = self.xx[best_pz]
        ## var_beta = self.denum*pzero_hat*(1-pzero_hat) # does'nt seem to work here 
        res_h1.update( { 'pzero' : self.xx[best_pz] } )
        lbd= (res_h1['llik']-res_h0['llik'])
        if lbd<0:
            lbd=np.nan
        return {'H0': res_h0, 'H1': res_h1 ,'Lbd': lbd, 'pval': chi2.sf(lbd,self.df)}

    def fit_loglik_optim_ML(self, p, skip_small=True):
        res_h0 = self.loglik_h0(p)
        pz=res_h0['pzero']
        if skip_small and (np.min([pz,1-pz]) < 0.05):
            res_h1={ 'pzero': np.nan, 'llik': np.nan, 'beta': np.full((self.df),np.nan)}
            lbd=np.nan
        else:
            val_h0,_ = root_pzero_h0( (p, res_h0['pzero'], self.Vinv))
            res_h0['pzero'] = val_h0
            res_h0['llik'] = -loglik_h0_qfunc( val_h0, p, self.Vinv)
            val_h1,_ = root_pzero( ( p, val_h0, self.xVinv,self.denum,self.x,self.Vinv))
            res_h1 = self.loglik_h1(val_h1,p)
            res_h1.update({ 'pzero' : val_h1} )
            lbd= (res_h1['llik']-res_h0['llik'])
            ##print(pz, res_h0['pzero'], res_h1['pzero'], res_h0['llik'], res_h1['llik'], file=self.logfile)
            # if correct:
            #     df = self.correct_lrt( res_h0['pzero'])
            #     lbd = chi2.ppf( chi2.cdf( lbd, df = df), df = self.df)
            ##print(res_h0['pzero'],val_h1.x[0],val_h1.success,val_h1.nfev,val_h1.nit,lbd)
        return {'H0': res_h0, 'H1': res_h1 ,'lrt': lbd, 'pvalue': chi2.sf(lbd,self.df)}

    def FLKadapt(self, frq, skip_small = True):
        try:
            npop, nsnp = frq.shape
        except:
            return self.fit_loglik_optim_ML(p, frq, skip_small)
        args=[]
        for isnp in range(nsnp):
                args.append((frq[:,isnp], self.xVinv, self.denum, self.x, self.Vinv, self.w))
        lrt_res = self.pool.map(calc_lrt, args)
        return lrt_res
  
    def fit_loglik_optim(self,p,skip_small=True, correct = True):
        ''' Numerical optimization of loglikelihood'''
        res_h0 = self.loglik_h0(p)
        pz=res_h0['pzero']
        if skip_small and (np.min([pz,1-pz]) < 0.05):
            res_h1={ 'pzero': np.nan, 'llik': np.nan, 'beta': np.full((self.df),np.nan)}
            lbd=np.nan
        else:
            val_h1 = optim(loglik_h1_qfunc,res_h0['pzero'], args=(p,self.xVinv,self.denum,self.x,self.Vinv),method='SLSQP', bounds = [ (0,1) ],options = {'maxiter':20})
            res_h1 = self.loglik_h1(val_h1.x[0],p)
            res_h1.update({ 'pzero' : val_h1.x[0]} )
            lbd= (res_h1['llik']-res_h0['llik'])
            if correct:
                df = self.correct_lrt( res_h0['pzero'])
                lbd = chi2.ppf( chi2.cdf( lbd, df = df), df = self.df)
            ##print(res_h0['pzero'],val_h1.x[0],val_h1.success,val_h1.nfev,val_h1.nit,lbd)
        return {'H0': res_h0, 'H1': res_h1 ,'Lbd': lbd, 'pvalue': chi2.sf(lbd,self.df)}

    #### hapFLKadapt
    def PCA_clust(self,kfrq):
        E,K,R = kfrq.shape
        Pbar = np.einsum('j,klj->kl',self.w,kfrq)
        Pbar=np.kron(Pbar,np.ones(R)).reshape(E,K,R)
        Pmat = np.concatenate(kfrq-Pbar).T
        assert Pmat.shape == (R,E*K)
        u,s,v = np.linalg.svd(Pmat)
        print('Global Lbd',*self.S)
        print('Local Lbd', *s)
        with open('test_Qglob.txt','w') as f:
            for r in range(R):
                print(*self.U[r,],file=f)
        with open('test_Qloc.txt','w') as f:
            for r in range(R):
                print(*u[r,],file=f)
        for pc in range(R):
            for r in range(R):
                print(pc,u[r,pc],self.U[r,pc],self.x[r,0],file=self.logfile)
    
    def BF_clust(self,kfrq, pmin = 0.05, beta_priors = None):
        if beta_priors is None:
            beta_priors = np.power( 10, np.linspace( -1, 0, 10))
        
        assert self.pool is not None
        E, K, R = kfrq.shape
        P = len( beta_priors)
        args = []
        Pbar = np.mean( kfrq, axis = 2)
        subset = np.array( ( Pbar > pmin) & ( Pbar < (1-pmin)))
        Keff = np.sum( subset, axis = 1)
        ## compute logBF_k for each EM 
        for e in range(E):
            for k in range(K):
                ## ajout boucle prior
                for ip, nu in enumerate( beta_priors):
                    self.set_beta_prior( var_prior = nu * self.diagFbar)
                    args.append( ( kfrq[e,k,], self.xx, self.logdet_omega, self.logdet_nu, self.logdet_V, self.beta_mean_fact, self.omega_inv, self.Vinv))
        logBFk =  np.array( self.pool.map( calc_bf, args))
        logBFk = logBFk.reshape( E, K, P)
        # print("New locus")
        # with open('bf.profile','w') as fbf:
        #     print('e','k','pz','nu','Keff','logbf',file=fbf)
        #     for e in range(E):
        #         for k in range(K):
        #             if pmin < Pbar[e,k] < (1 - pmin) :
        #                 for ip, nu in enumerate( beta_priors):
        #                     print( e, k, Pbar[ e, k], nu, Keff[e], logBFk[ e, k, ip],file=fbf)
                    
        ## only consider clusters which pmin < mean(pk) < pmin
        logBFe = np.zeros(E)
        for e in range(E):
            subK = range(K)
            for k in subK:
                if pmin < Pbar[ e, k] < (1-pmin):
                    logbfk_i =  average_log10bf( logBFk[ e, k, ])
                    logBFe[e] += logbfk_i #average_log10bf( logBFk[ e, k, ] )
            ##print( e, logBFe[ e])
            # if Keff > 0:
            #     ## uniform prior on number of associated clusters
            #     logBFe[e] += Keff * np.log10( (Keff + 1)/( 2*Keff))
        return average_log10bf( logBFe)

    
    def hapflkadapt(self, kfrq):
        assert self.pool is not None
        assert self.correct_lrt is not None
     
        if len(kfrq.shape)==4:
            E, K, R, L = kfrq.shape
        elif len(kfrq.shape) == 3:
            return self.fit_loglik_clust_pz(kfrq)
        else:
            raise ValueError("Bad Shape for kfrq")
        
        LRT  = np.zeros( ( E, K, L))
        Succ = np.zeros( ( E, K, L))
        args = []
        for e in range(E):
            for k in range(K):
                for l in range(L):
                    args.append( ( kfrq[e,k,:,l], self.xVinv, self.denum, self.x, self.Vinv, self.w))
        res =  self.pool.map( calc_lrt, args)
        
        LRT = np.array( [ x['lrt'] for x in res])
        Succ = np.array([ x['pzsucc'] for x in res])
        
        LRT = LRT.reshape(E,K,L)
        Succ = Succ.reshape(E,K,L)
        LRT = LRT*Succ

        hflka_res = []
        for l in range(L):
            myLRT = LRT[:,:,l]
            mykfrq = kfrq[:,:,:,l]
            Pbar = np.einsum('j,klj->kl',self.w,mykfrq)
            subset = np.array( ( Pbar > 0.05) & ( Pbar < 0.95))
            df_pbar = self.correct_lrt(Pbar)
            df = np.sum( df_pbar*subset , axis=1)
            lrt = np.sum( myLRT*subset, axis=1)
            qchisq = chi2.sf(lrt, df)
            zscores = -norm.ppf( qchisq)
            Ztot = np.mean(zscores)
            hflka_res.append({ 'Z':Ztot, 'pvalue': norm.sf(Ztot), 'Z_mat': zscores})
        return hflka_res

    def fit_loglik_clust_pz(self,kfrq):
        '''
        Computes Likelihood Ratio Test of model:

        kfrq = p0_k + X beta_k + e 
        e ~ N( 0, F. p0_k_1*(1-p0_k_1))

        against:

        kfrq = p0_k + e 
        e ~ N( 0, F. p0_k_0*(1-p0_k_0))
        input:
            -- kfrq: cluster frequencies ( E x K x R ) 
            with E (Number of EM fits), K (number of clusters), R (number of pops)
        '''
        assert self.pool is not None
        assert self.correct_lrt is not None
        E, K, R = kfrq.shape
        args = []
        for e in range(E):
            for k in range(K):
                args.append( ( kfrq[e,k,], self.xVinv, self.denum, self.x, self.Vinv, self.w))
        res =  self.pool.map( calc_lrt, args)
        LRT = np.array( [ x['lrt'] for x in res])
        Succ = np.array([ x['pzsucc'] for x in res])

        LRT = LRT.reshape(E,K)
        Succ = Succ.reshape(E,K)
            
        LRT = LRT*Succ
        
        Pbar = np.einsum('j,klj->kl',self.w,kfrq)
        subset = np.array( ( Pbar > 0.05) & ( Pbar < 0.95))
        df_pbar = self.correct_lrt(Pbar)
        df = np.sum( df_pbar*subset , axis=1)
        # df = np.sum( subset , axis=1)
        lrt = np.sum( LRT*subset, axis=1)
        qchisq = chi2.sf(lrt, df)
        zscores = -norm.ppf( qchisq)
        Ztot = np.mean(zscores)
        return({ 'Z':Ztot, 'pvalue': norm.sf(Ztot), 'Z_mat': zscores})
  
    def fit_loglik_clust(self,kfrq):
        '''
        Computes Likelihood Ratio Test of model :

        kfrq = p0_k + X beta_k + e 
        e ~ N( 0, F . var_k_1)
        
        against:
        
        kfrq = p0_k + e
        e ~ N(0, F . var_k_0)

        input:
        -- kfrq: cluster frequencies ( E x K x R ) 
        with E (Number of EM fits), K (number of clusters), R (number of pops)
        '''
        E,K,R = kfrq.shape
        P = self.df
        pk = kfrq.reshape( ( E, K, R, 1))
        ## MLE  under H1
        unx = np.hstack( (self.un.reshape(R,1), self.x))
        ### ( X' . Vinv . X)inv . (X' . Vinv) p 
        Emat =  np.linalg.inv( multi_dot( [unx.T, self.Vinv, unx]) )
        Emat = multi_dot( [ Emat, unx.T, self.Vinv] )
        mles = np.einsum('ij,kljm->klim',Emat,pk)
        ### yhat = X . B
        fitted = np.einsum('ij,kljm->klim',unx,mles)
        resid = pk - fitted
        ### (y - yhat)' Vinv (y - yhat)
        RSS = np.einsum('ij,kljm->klim',self.Vinv,resid)
        RSS = np.einsum('ijkl,ijkm->ijlm',resid,RSS)
        mle_sigma=RSS/R
        ll_h1 = -R*np.log(mle_sigma) - RSS/mle_sigma
        ## MLE under H0
        _un = self.un.reshape(R,1)
        Emat = np.linalg.inv( multi_dot( [_un.T, self.Vinv, _un]) )
        Emat = multi_dot( [ Emat, _un.T, self.Vinv] )
        mles_H0 = np.einsum('ij,kljm->klim',Emat,pk)
        fitted_H0 = np.einsum('ij,kljm->klim',_un,mles_H0)
        resid_H0 = pk - fitted_H0
        RSS_H0 = np.einsum('ij,kljm->klim',self.Vinv,resid_H0)
        RSS_H0 = np.einsum('ijkl,ijkm->ijlm',resid_H0,RSS_H0)
        mle_sigma_H0 = RSS_H0/R
        ll_h0 = -R*np.log(mle_sigma_H0) - RSS_H0/mle_sigma_H0

        LRT = (ll_h1 - ll_h0).reshape(E,K)
        LRT_av = np.mean(np.sum(LRT,axis=1))
        return({ 'Lbd':LRT_av, 'pvalue': chi2.sf(LRT_av,df=K), 'LRT_mat': LRT})
        
    ## Bayes Factors
    def set_beta_prior(self,var_prior=0.1):
        self.prior = True
        self.nu = var_prior
        ## 1 par
        self.nuinv = np.diag([1/self.nu]*self.df)
        self.logdet_nu = self.df * np.log(self.nu) 
        self.omega_inv = self.nuinv + self.xVinvx
        self.omega = np.linalg.inv(self.omega_inv)
        self.beta_mean_fact = np.dot( self.omega, self.xVinv)
        _,self.logdet_omega = np.linalg.slogdet( self.omega)
   
    def bayes_factor_vect(self,p):
        if len(p.shape) == 1:
            frq = p.reshape( self.npop, 1)
        else: 
            frq = p
        ns = frq.shape[1]
        pzero = np.broadcast_to( self.xx, ( ns, len(self.xx)) )
        
        ## scaling factor
        lp = - self.npop * np.log( pzero * (1-pzero))
        lp_1 = lp + self.logdet_omega - self.logdet_nu - self.logdet_V
        lp_0 = lp - self.logdet_V
        ## SNP Sum of Squares
        res = (frq - pzero).T
        res=res.reshape( len(self.xx), self.npop, ns)
        postmean = np.einsum( 'ij,kjl->kil', self.beta_mean_fact, res)
        betass = np.einsum( 'ij,kjl->kil', self.omega_inv, postmean)
        BT = np.einsum('ijk->ikj',postmean)
        betass = np.einsum( 'ijk,ikj->i',BT,betass)
        datass=np.einsum('ij,kjl->kil',self.Vinv,res)
        RT=np.einsum('ijk->ikj',res)
        datass=np.einsum('ijk,ikj->i',RT,datass)
        ## compute Bayes Factor
        lp_0 -= datass / (pzero * (1-pzero))
        lp_1 -= ( datass - betass) / ( pzero * ( 1 - pzero))
        vmax0 = np.max(lp_0)
        P0 = np.log( np.sum( np.exp( lp_0 - vmax0) ) ) + vmax0
        vmax1 = np.max(lp_1)
        P1 = np.log( np.sum( np.exp( lp_1 - vmax1) ) ) + vmax1
        return (P1-P0)/np.log(10)

    def bayes_factor(self,p, beta_priors = None ):
        ## priors: variance of divergence in scale of drift units
        ## assumes covariables are standardized.
        if beta_priors is None:
            beta_priors = np.power(10,np.linspace(-1,1,10))
        bf = np.zeros(len(beta_priors))
        for i,nu in enumerate(beta_priors):
            self.set_beta_prior( var_prior = nu * self.diagFbar )
            bf[i] =  self.bayes_factor_vect(p)
        vmax = np.max(bf)
        bf = np.log10( np.sum( np.power(10, bf - vmax))) + vmax
        return bf - np.log10( len(beta_priors))

    def bayesFLKadapt( self, frq, beta_priors = np.power( 10, np.linspace(-1,1,10) )):
        npop,nsnp = frq.shape
        P = len(beta_priors)
        args = []
        for ip, nu in enumerate( beta_priors):
            self.set_beta_prior( var_prior = nu*self.diagFbar)
            for s in range(nsnp):
                args.append( (frq[:, s], self.xx, self.logdet_omega, self.logdet_nu, self.logdet_V, self.beta_mean_fact, self.omega_inv, self.Vinv))
        logBF_t = np.array( self.pool.map( calc_bf, args))
        logBF_t = logBF_t.reshape( P, nsnp)
        logBF = self.pool.map( average_log10bf, logBF_t.T)
        return logBF
            
        
    def bf_rank(self,val):
        assert self.pool is not None
        val = np.asarray(val)
        args=[]
        for x in val:
            args.append((self.bf_null,x))
        scores = self.pool.map(pscore, args)
        scores = np.array( scores )
        p = 1 - scores/100
        p[ p==0] = 0.5/len(self.bf_null)
        p[ p==1] =  1-0.5/len(self.bf_null)
        return p
        
    def logprob(self,pzero,p):
        ''' Computes log(p|pzero,x,Beta) under Beta=0 and Beta>0 
        Returns:
            lp = [logp0,logp1] 
        '''
        ## scaling factor
        lp = - self.npop * np.log( pzero * (1-pzero))
        lp_1 = lp + self.logdet_omega - self.logdet_nu - self.logdet_V
        lp_0 = lp - self.logdet_V

        ## SNP Sum of Squares
        res = (p - pzero)
        postmean = np.dot( self.beta_mean_fact, res )
        betass = multi_dot( ( postmean.T, self.omega_inv, postmean) )
        datass = multi_dot( ( res.T, self.Vinv, res ) )

        lp_0 -= datass / ( pzero * ( 1 - pzero))
        lp_1 -= ( datass - betass) / ( pzero * ( 1 - pzero))
        return np.array([lp_0,lp_1])

 
        
# if __name__ == '__main__':    
#     #myh=HapFLK.from_db_file('../hapflkadapt/covkin.db')
#     ##myh=HapFLK.from_db_file('./hapflk.db')
#     #myh=HapFLK.from_db_file('../hapflkadapt/bta29.db')
#     ##myh.kinship=myh.kinship+0.01*(np.trace(myh.kinship)/myh.npop)

    
#     # input_file='../hapflkadapt/bta29.db'
#     try:
#         input_file=sys.argv[1]
#         output_file_prefix=input_file[:-3]+'_test'
#     except IndexError:
#         ##input_file='../hapflkadapt/BTA2.db'
#         input_file='../hapflkadapt/bta29.db'
#         output_file_prefix='test'
#         print(input_file,output_file_prefix)
#         myh=HapFLK.from_db_file(input_file)
#         nullcov = norm.rvs(size=myh.npop)
#         myh.counter.new('NULL Analysis')
#         Covs=np.vstack([nullcov]).T
#         with FLKadapt(myh.kinship, Covs, prefix='test',ncpu=16) as ffnull:
#             ##ffnull.calibrate_null(nsim=1000)
#             test_snps = range(0,myh.nsnp,1)
#             myh.counter.new('hapFLKadapt')
#             if myh.kfrq is not None:
#                 #pass
#                 # resnull_h = [ffnull.PCA_clust(myh.kfrq[:,:,:,i]) for i in range(200,myh.nsnp,20000)]
#                 resnull_h = [ffnull.fit_loglik_clust_pz(myh.kfrq[:,:,:,i]) for i in test_snps]
#                 # resnull_h = [ffnull.BF_clust(myh.kfrq[:,:,:,i]) for i in range(0,myh.nsnp,200)]
#             # myh.counter.new('FLKadapt: BF Stat 2 pars')
#             # bfnull2 = np.apply_along_axis(ffnull.bayes_factor_2pars,0,myh.frq)
#             # myh.counter.new('LRT raw')
#             # lrtraw = np.apply_along_axis(ffnull.fit_loglik_optim,0,myh.frq, correct = False)
#             # myh.counter.new('LRT cor')
#             # lrtcor = np.apply_along_axis(ffnull.fit_loglik_optim,0,myh.frq)
#             myh.counter.new('LRT ML')
#             lrtml = np.apply_along_axis(ffnull.fit_loglik_optim_ML,0,myh.frq)
#             myh.counter.new('BF')
#             bfnull  = np.apply_along_axis(ffnull.bayes_factor,0,myh.frq)
#             # print(bfnull2.shape,lrtnull.shape,bfnull.shape)
#             with open( 'test.out', 'w') as fout:
#                 print( 'LRTr', 'pvalr', 'LRTc', 'pvalc', 'LRTML','pvalML', 'bf',  'Zh', 'pvalh', file=fout)
#                 for ix,i in enumerate(test_snps):
#                     print( lrtraw[i]['Lbd'], lrtraw[i]['pvalue'], lrtcor[i]['Lbd'], lrtcor[i]['pvalue'], lrtml[i]['Lbd'], lrtml[i]['pvalue'], bfnull[i], resnull_h[ix]['Z'], resnull_h[ix]['pvalue'], file=fout)
#         myh.counter.end()
#         sys.exit(0)
            
#     print(input_file,output_file_prefix)
#     myh=HapFLK.from_db_file(input_file)
    
#     covdata=pd.read_table('../data/breed_phenotypes.txt',sep=' ')
#     covpops=list(covdata.index)
#     smscov=np.array( [ covdata.values[ covpops.index(p), 1] for p in myh.pops])
#     smscov=(smscov-np.mean(smscov))/ np.std(smscov)
#     prodcov=np.array( [ covdata.values[ covpops.index(p), 0] == 'Dairy' for p in myh.pops], dtype=np.int)
    
#     ## Stature
#     myh.counter.new('Stature Analysis')
#     Covs=np.vstack([smscov]).T
#     with FLKadapt(myh.kinship, Covs, prefix = output_file_prefix, ncpu=16) as ffsms:
#         myh.counter.new('Null Simulations')
#         ffsms.calibrate_null()
#         myh.counter.new('FLKadapt: LRT Stat')
#         ressms = np.apply_along_axis(ffsms.fit_loglik_optim_ML,0,myh.frq)
#         myh.counter.new('FLKadapt: BF Stat')
#         bfsms = np.apply_along_axis(ffsms.bayes_factor,0,myh.frq)
#         rankbfsms = ffsms.bf_rank(bfsms)
#         if myh.kfrq is not None:
#             myh.counter.new('hapFLKadapt')
#             ressms_h = [ffsms.fit_loglik_clust_pz(myh.kfrq[:,:,:,i]) for i in range(myh.nsnp)]
         
#     ## Write out
#     myh.counter.new('Writing results')
#     with open(output_file_prefix+'_sms_opt.out','w') as fout:
#         print('rs','chr','pos','pzero.null','bf', 'bfrank', 'LRT', 'pvalue', 'Zh', 'pvalueh', 'pzero','beta', file=fout)
#         for i,d in enumerate(ressms):
#             rs=myh.sorted_snps[i].name
#             chrom,_,pos=myh.carte.position(rs)
#             print( rs, chrom, int(pos), d['H0']['pzero'], bfsms[i], rankbfsms[i],  d['Lbd'],  d['pvalue'], ressms_h[i]['Z'], ressms_h[i]['pvalue'], d['H1']['pzero'], *d['H1']['beta'],  file=fout)

#     ## Production
#     myh.counter.new('Type Analysis')
#     Covs=np.vstack([prodcov]).T
#     with FLKadapt(myh.kinship, Covs, prefix = output_file_prefix, ncpu=16) as ffprod:
#         myh.counter.new('Null Simulations')
#         ffprod.calibrate_null()
#         myh.counter.new('FLKadapt: LRT Stat')
#         resprod = np.apply_along_axis( ffprod.fit_loglik_optim_ML, 0, myh.frq)
#         myh.counter.new('FLKadapt: BF Stat')
#         bfprod = np.apply_along_axis( ffprod.bayes_factor, 0, myh.frq)
#         rankbfprod = ffprod.bf_rank(bfprod)
#         if myh.kfrq is not None:
#             myh.counter.new('hapFLKadapt')
#             resprod_h = [ffprod.fit_loglik_clust_pz(myh.kfrq[:,:,:,i]) for i in range(myh.nsnp)]

  
#     myh.counter.new('Writing results')
#     with open(output_file_prefix+'_prod_opt.out','w') as fout:
#         print('rs','chr','pos','pzero.null','bf', 'bfrank', 'LRT', 'pvalue', 'Zh', 'pvalueh', 'pzero','beta', file=fout)
#         for i,d in enumerate(resprod):
#             rs=myh.sorted_snps[i].name
#             chrom,_,pos=myh.carte.position(rs)
#             print( rs, chrom, int(pos), d['H0']['pzero'], bfprod[i], rankbfprod[i], d['Lbd'],  d['pvalue'], resprod_h[i]['Z'], resprod_h[i]['pvalue'], d['H1']['pzero'], *d['H1']['beta'],  file=fout)
#     print()
 
#     myh.counter.end()
