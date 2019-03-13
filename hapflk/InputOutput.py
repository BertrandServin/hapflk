import sys
import re
import argparse
import numpy as np
from bisect import insort
from hapflk import missing
from hapflk import data
import hapflk._pgenlib as _pgenlib

def plink_bfile(string):
    try:
        open(string+'.bed')
    except IOError:
        msg="can't open BED file %s"%string+'.bed'
        raise argparse.ArgumentTypeError(msg)
    try:
        open(string+'.fam')
    except IOError:
        msg="can't open FAM file %s"%string+'.fam'
        raise argparse.ArgumentTypeError(msg)
    try:
        open(string+'.bim')
    except IOError:
        msg="can't open BIM file %s"%string+'.bim'
        raise argparse.ArgumentTypeError(msg)
    return string

def shapeit_file(string):
    try:
        open(string+'.haps')
    except IOError:
        msg="can't open HAPS file%s"%string+'.haps'
        raise argparse.ArgumentTypeError(msg)
    try:
        open(string+'.sample')
    except IOError:
        msg="can't open SAMPLE file%s"%string+'.sample'
        raise argparse.ArgumentTypeError(msg)
    return string

io_parser=argparse.ArgumentParser(add_help=False,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
input_group=io_parser.add_argument_group('Input Files')
input_type_group = input_group.add_mutually_exclusive_group(required=True)
input_type_group.add_argument('--bfile',metavar='PREFIX',help='PLINK bfile prefix (bim,fam,bed)',type=plink_bfile)
input_type_group.add_argument('--sfile',metavar='PREFIX',help='ShapeIT file prefix (haps,sample)',type=shapeit_file)


defaultParams= {
    "MissGeno":'0',
    "MissPheno":'-9',
    "MissParent":'0',
    "MissSex":'0'
    }


key="kshuwewekiuvf"

def _get_snpIdx(myMap,chrom,left,right,othermap=False):
    '''
    Return index of SNPs in map

    Parameters
    --------------
    myMap : the map
    chrom : chromosome
    left : leftmost position
    right : rightmost position
    othermap : use other (genetic / RH) map scale
    '''
    if othermap:
        getMap=myMap.genetMap
    else:
        getMap=myMap.physMap
    mySnps=[ x for x in getMap(chrom,xleft=left,xright=right)]
    mySnpIdx=[]
    for i,s in enumerate(myMap.input_order):
        if s in mySnps:
            mySnpIdx.append(i)
    return mySnpIdx

##### generic input function ####
def parseInput(options,params=defaultParams):
    '''
    Read input files 
    
    Parameters
    --------------
    options : options from io_parser
    params : parameters for coding used in input files, overidden using options

    Returns
    ----------
    res: dictionary with keys:
         -- dataset : an instance from the gwas Dataset class
         -- map : an instance from the gwas Map class

    See also
    ----------
    data.Dataset
    data.Map
    '''

    ## read input
    if options.bfile:
        return parsePlinkBfile(options.bfile,params=params,options=options)
    elif options.sfile:
        return parseShapeIT(options.sfile, params=params)
        
               
################ SHAPEIT2 files ####################################
def parseShapeIT(prefix, params=defaultParams):
    ## parse sample file
    idata=[]
    with open(prefix+'.sample') as f:
        for il,ligne in enumerate(f):
            if il<2:
                continue
            buf=ligne.split()
            phe=((buf[6]!=params['MissPheno']) and float(buf[6])) or None
            idata.append([buf[0], \
                          buf[1],  \
                              ((buf[3]!=params['MissParent']) and buf[3]) or None, \
                              ((buf[4]!=params['MissParent']) and buf[4]) or None, \
                              (buf[5]!=params['MissSex'] and buf[5]) or None, \
                              phe])
                              
    ## parse haps file
    SNPs=[]
    myMap=data.Map()
    hdata = []
    with open(prefix+'.haps') as f:
        for ligne in f:
            buf=ligne.split()
            hdata.append( [ int(i) for i in buf[5:] ] )
            ## create SNP
            myS=data.SNP(buf[1])
            myS.alleles[0]=buf[3]
            myS.alleles[1]=buf[4]
            ## add marker to map
            try:
                mychr=int(buf[0])
            except:
                mychr=buf[0]
            mychr=buf[0]
            myMap.addMarker(M=buf[1],C=mychr,posG=float(buf[2])*1e-6,posP=float(buf[2]))
            SNPs.append(myS)
    sdata = {'snps':SNPs, 'map':myMap}
    ni = 2*len(idata)
    ns = len(sdata['snps'])
    mySnpIdx = range(ns)
    dataset = data.Dataset(prefix,nsnp = ns, nindiv = ni)
    ## fill in SNP data
    for s in [sdata['snps'][i] for i in mySnpIdx]:
        dataset.addSnp(s.name)
        dataset.snp[s.name].initAlleles(s.alleles[0],s.alleles[1])
    for iind,ind in enumerate(idata):
        dataset.addIndividual(pop=ind[0],
                              ID=ind[1]+'.1',
                              fatherID=ind[2],
                              motherID=ind[3],
                              sex=ind[4],
                              phenotype=ind[5])
        dataset.addIndividual(pop=ind[0],
                              ID=ind[1]+'.2',
                              fatherID=ind[2],
                              motherID=ind[3],
                              sex=ind[4],
                              phenotype=ind[5])
        dataset.Data[2*iind  ,:]=np.array( [ 2*hdata[s][2*iind] for s in mySnpIdx], dtype=np.int)
        dataset.Data[2*iind+1,:]=np.array( [ 2*hdata[s][2*iind+1] for s in mySnpIdx], dtype=np.int)
    return {'dataset':dataset, 'map': sdata['map']}

################ BED / BIM / FAM files #############################

def parsePlinkBfile(prefix,noPheno=False,params=defaultParams,options=None):
    fam=prefix+'.fam'
    bim=prefix+'.bim'
    bed=prefix+'.bed'
    ## individual data
    idata=parseFamFile(fam) ## options not implemented
    ni=len(idata)
    ### SNP data
    sdata=parseBimFile(bim)
    ns=len(sdata['snps'])
    mySnpIdx=range(ns)
    dataset=data.Dataset(prefix,nsnp=ns,nindiv=ni)
    ## fill in SNP data
    for s in [sdata['snps'][i] for i in mySnpIdx]:
        dataset.addSnp(s.name)
        dataset.snp[s.name].initAlleles(s.alleles[0],s.alleles[1])
    ## fill in indiv data
    for ind in idata:
        dataset.addIndividual(pop=ind[0],
                              ID=ind[1],
                              fatherID=ind[2],
                              motherID=ind[3],
                              sex=ind[4],
                              phenotype=ind[5])
    fillBedData_fast(bed.encode(),dataset.Data)
    return {'dataset':dataset,'map':sdata['map']}

def parseFamFile(fileName,noPheno=False,params=defaultParams):
    indiv_data=[]
    with open(fileName) as f:
        for ligne in f:
            buf=ligne.split()
            if noPheno:
                phe=None
            else:
                phe=((buf[5]!=params['MissPheno']) and float(buf[5])) or None
  
            indiv_data.append([buf[0], \
                               buf[1],  \
                               ((buf[2]!=params['MissParent']) and buf[2]) or None, \
                               ((buf[3]!=params['MissParent']) and buf[3]) or None, \
                               (buf[4]!=params['MissSex'] and buf[4]) or None, \
                                phe])
    return indiv_data

def parseBimFile(fileName):
    SNPs=[]
    myMap=data.Map()
    with open(fileName) as f:
        for ligne in f:
            buf=ligne.split()
            ## create SNP
            myS=data.SNP(buf[1])
            myS.alleles[0]=buf[4]
            myS.alleles[1]=buf[5]
            ## add marker to map
            try:
                mychr=int(buf[0])
            except:
                mychr=buf[0]
            mychr=buf[0]
            myMap.addMarker(M=buf[1],C=mychr,posG=float(buf[2]),posP=float(buf[3]))
            SNPs.append(myS)
    return {'snps':SNPs,'map':myMap}

def fillBedData_fast(fileName,DataMatrix):
    n_indiv,n_snp=DataMatrix.shape
    reader=_pgenlib.PgenReader(fileName,raw_sample_ct=n_indiv)
    buf=np.empty(n_indiv,np.int8)
    for isnp in range(n_snp):
        reader.read(isnp,buf,0)
        buf[buf==-9]=missing
        DataMatrix[:,isnp]=buf

def fillBedData_slow(fileName,DataMatrix,snpidx=None):
    n_indiv,n_snp=DataMatrix.shape
    ## use dict type for faster look up
    ## key index in total, val : index in DataMatrix
    if snpidx is None:
        snpidx=dict([ (i,i) for i in range(n_snp)])
    else:
        snpidx=dict([ (s,i) for i,s in enumerate(snpidx)])
    cur_snp=0
    ## bit enumerator
    def bits(f):
        bytes = (ord(b) for b in f.read())
        for b in bytes:
            for i in range(8):
                yield (b >> i) & 1
    ## bit pair to geno converter
    def bpair_2_geno(pair):
        '''
        from plink doc
        '''
        if pair=='00':
            return 0
        if pair=='11':
            return 2
        if pair=='01':
            return 1
        if pair=='10':
            return missing
    ## constants
    target_magic='0011011011011000'
    snp_major='10000000'    
    indiv_major='00000000'
    ### let's go
    bed_stream=open(fileName, 'r')
    magic=[]
    f_mode=[]
    i_major=0
    i_minor=0
    proceed_2_next_bit=False
    bit_geno=[]
    nbit=-1
    for b in bits(bed_stream):
        nbit+=1
        ## Test for magic number
        if nbit<16:
            magic.append(str(b))
            continue
        if nbit==16:
            magic=''.join(magic)
            if magic!=target_magic:
                print('Not a bed file')
                raise ValueError
        ## Find out packing mode
        if nbit<24:
            f_mode.append(str(b))
            continue
        if nbit==24:
            f_mode=''.join(f_mode)
            if f_mode==snp_major:
                pack_len=n_indiv 
            elif f_mode==indiv_major:
                pack_len=n_snp
            else:
                print('Cannot determine packing mode, abort')
                raise ValueError
        ## Now i>24, we are reading genotypes
        ## case where we just wait for next bit to start reading again
        if proceed_2_next_bit:
            if nbit%8!=0:
                continue
            else:
                proceed_2_next_bit=False
        ## reading genotypes        
        bit_geno.append(str(b))
        if nbit%2!=0:
            ##assert len(bit_geno)==2
            ## print nbit,i_major,i_minor,bit_geno
            geno=bpair_2_geno(''.join(bit_geno))
            if f_mode==snp_major:
                try:
                    data_idx=snpidx[i_major]
                    DataMatrix[i_minor,data_idx]=geno
                except KeyError:
                    pass
            else:
                try:
                    data_idx=snpidx[i_minor]
                    DataMatrix[i_major,data_idx]=geno
                except KeyError:
                    pass
            i_minor += 1 
            bit_geno=[]
        if i_minor==pack_len:
            proceed_2_next_bit=True
            i_minor=0
            i_major+=1
        

############################ Covariate Information Files ##################

## File Format

## Header :ID cov1_name cov2_name ...
## Indiv1/pop1 val1 val2 ...

## add parser options to read in covariate files
def populate_covariate_args(parser):
    ''' Add options to read covariates in option parser'''
    covariate_args=parser.add_argument_group('Covariates','')
    covariate_args.add_argument('--covar',dest='covar_file',metavar='FILE',default=None,
                                help='File with available covariates. Use --cov or --qcov to include them in the analysis')
    covariate_args.add_argument('--cov',metavar='LIST',dest='cov',default='',
                                help='List of *qualitative* covariates to include, separated by commas')
    covariate_args.add_argument('--qcov',metavar='LIST',dest='qcov',default='',
                                help='List of *quantitative*  covariates to include, separated by commas')

def get_covariates_raw(fichier,missing_value='NA'):
    ''' Get raw data from covariate file /fichier/

    File Format:
    Header     : ID cov1_name cov2_name ...
    Other lines: ID1 val1_1 val1_2 ...

    Returns a dictionary with covariate names as keys (read from header)
    For each covariate, the value is a dictionary with IDs as keys and string as value
    No conversion is done. Missing value key can be specified (default = NA)
    '''
    f=open(fichier)
    covar_dat=f.readlines()
    covar_name=covar_dat[0].split()[1:]
    covariates={}
    for nom in covar_name:
        covariates[nom]={}
    for ligne in covar_dat[1:]:
        buf=ligne.split()
        indiv=buf[0]
        for i,covar in enumerate(covar_name):
            if buf[i+1]==missing_value:
                continue
            covariates[covar][indiv]=buf[i+1]## no conversion done here
    return covariates

def get_covariates_matrix(filename,fact_names,qcov_names,names,stdize=True):
    ''' 
    Returns a design matrix of all covariates listed in fact_names and qcov_names. 
    fact_names, qcov_names: string of qualitative (resp. quantitative) covariates, separated by commas.
    names : list of ID names providing the required ordering of rows.
    stdize : if True, standardize quantitative covariates (X-m)/s.

    Returns a dictionary with keys:
    -- DesignMatrices: dict with covar names as keys and DM as value
    -- levels: dict with cofact names as keys and ordered levels as value
    '''
    covariates=get_covariates_raw(filename)
    cofact=[] ## qualitative cofactors
    covar=[] ## quantitative covariables
    nids=len(names)
    try:
        for c in fact_names.split(','):
            if c!='':
                if c in covariates:
                    cofact.append(c)
                else:
                    print( 'Covariate %s not found in covariate file, ignoring.'%c)
    except:
        pass
    try:
        for c in qcov_names.split(','):
            if c!='':
                if c in covariates:
                    covar.append(c)
                else:
                    print( 'Covariate %s not found in covariate file, ignoring.'%c)
    except:
        pass
    Matrices={}
    levels={}
    for q in cofact:
        q_vec=[]
        for nom in names:
            q_vec.append(covariates[q][nom])
        u,aa=np.unique(q_vec,return_inverse=True)
        print( 'Cofactor "%s" with %d levels'%(q,len(u)))
        print( 'Levels:' *u)
        levels[q]=u
        M=np.zeros((nids,len(u)),dtype=np.int)
        M[range(nids),aa]=1
        Matrices[q]=M[:,1:]
    for q in covar:
        M=np.zeros((nids,1),dtype=np.float)
        for i,nom in enumerate(names):
            try:
                val=float(covariates[q][nom])
                M[i,0]=val
            except ValueError:
                raise ValueError('Covariate %s is not quantitative !'%q)
        mu=np.mean(M)
        ss=np.std(M)
        if stdize:
            M=(M-mu)/ss
        print( 'Covariate "%s" with mean %f and sd %f'%(q,mu,ss))
        Matrices[q]=M
    return { 'cofactors' : cofact,
                 'covariables' : covar,
                 "DesignMatrices":Matrices,
                 "factor_levels":levels}
