#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Purpose:
  Run fixed effects, pooled effects and random estimation as in Tab 17.2 of Hansen 19/8


Date:
  19/10/12

Author:
    Aytek Mutlu
"""
###########################################################
### Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Additional routine, easy latex matrix output
# from lib.printtex import *

###########################################################
### mA= CreateDum(vA)
def CreateDum(vA):
    """
    Purpose:
      Create dummy matrix for all industries
    """
    vAu= np.unique(vA)
    iO= vA.shape[0]
    iA= vAu.shape[0]
    mA= np.zeros((iO, iA))
    for (i, a) in enumerate(vAu):
        vI= vA == a
        mA[vI,i]= 1

    return mA

###########################################################
### [vId, vY, mX, vA]= InitData(sIn, sId, sIC, sY, asX)
def InitData(sIn, sId, sIC, sY, asX):
    """
    Purpose:
        Read the prepared data file

    Inputs:
        sIn     string, data file
        sId     string, id indicator
        sIC     string, industry code indicator
        sY      string, y-variable
        asX     iK list of strings, x-variables

    Return value
        vId      iO vector, ID of company
        vY       iO vector, dependents
        mX       iO x iK matrix, independents, excluding constant
        vA       iO vector, industry codes
    """
    # Read the necessary data, grouped by year and id
    cols = ['year', sId, sY, sIC]+asX
    df= pd.read_excel(sIn)
    df = df[cols]
    df.set_index(['year', sId],inplace=True)
    # Lag the x-variables (assuming nyseamex is constant over the full period for a company...)
    dfL= df[asX].groupby(level=sId).shift(1)
    # dfL= dfL.rename(columns=lambda x: x+'L')          # Do not rename the X-variables


    df2= dfL.join(df[[sY, sIC]])           # Put the y-variable back in, with industry codes
    df= df[[sY, sIC]+asX]                  # Re-order variables
    df2= df2[[sY, sIC]+asX].dropna()       # Re-order, and get rid of missings

    # Check that indeed the x-s got lagged, and y is put in place
    # df.head(20)
    # df2.head(20)

    # Extract year/id from index, which now is a list of tuples
    vT= np.array([vI[0] for vI in df2.index])       # Get years from the tuples
    vId= np.array([vI[1] for vI in df2.index])      # Get ID from tuples
    vA= df2[sIC].values

    # Select y and X
    vY= df2[sY].values
    mX= df2[asX].values

    # Get industry dummies
    mA= CreateDum(vA)

    return (vId, vT, vY, mX, mA)

def Pooled(mX,mA,vY):
    """
    Purpose:
        calculate Pooled estimators

    Inputs:
        mX              iO x iK matrix, independents, excluding constant
        mA              iO x iA matrix, dummies
        vY              iO vector, dependents

    Return value
        params          4x1 vector, model coefficients
        standard errors 4x1 vector, model standard errors

    """
    
    mX_pooled = np.concatenate([mX,mA],axis=1)
    mX_pooled = sm.add_constant(mX_pooled)
    model_pooled = sm.OLS(vY,mX_pooled).fit(cov_type='HC1')
    print(model_pooled.summary())
    return model_pooled.params[2:6],model_pooled.bse[2:6]

def Fixed(mX,vY,vId,iO,iN):
        """
    Purpose:
        calculate Fixed estimators

    Inputs:
        mX              iO x iK matrix, independents, excluding constant
        vY              iO vector, dependents
        vId             iO vector, ID of company
        iO              Integer, total number of observations
        iN              Integer, total number of groups

    Return value
        params          3x1 vector, model coefficients
        standard errors 3x1 vector, model standard errors

    """
    
    
    fixed_data = np.concatenate([vY.reshape(-1,1),mX],axis=1)
    fixed_data = pd.DataFrame(np.concatenate([vId.reshape(-1,1),fixed_data],axis=1))
    fixed_data_demeaned = fixed_data.groupby(0).apply(lambda x: x - np.mean(x,axis=0))
    fixed_data_demeaned.drop(columns=[0],inplace=True)
    
    mX_fixed  = fixed_data_demeaned[[2,3,4]]
    mY_fixed  = fixed_data_demeaned[1]
    fixed = sm.OLS(mY_fixed,mX_fixed)
    model_fixed = fixed.fit(cov_type='HC1')
    print(model_fixed.summary())
    model_fixed_std_err = model_fixed.bse * np. sqrt ((iO - mX_fixed.shape[1])/( iO - (mX_fixed.shape[1]+iN)))
    print(model_fixed_std_err) 
    return model_fixed.params,model_fixed_std_err

def Random(mX,vY,vT,vId,iN,iT):
            """
    Purpose:
        calculate Random estimators

    Inputs:
        mX              iO x iK matrix, independents, excluding constant
        vY              iO vector, dependents
        vT              iO vector, years of observations
        vId             iO vector, ID of company
        iT              Integer, total number of distinct years
        iN              Integer, total number of groups

    Return value
        params          4x1 vector, model coefficients
    """
    
    
    
    ##concatenate all data
    mX_random = np.concatenate([vId.reshape(-1,1),vT.reshape(-1,1),vY.reshape(-1,1),mX],axis=1)
    full_data  = np.zeros((iN*iT,2))
    
    #handle unbalanced panel
    for i in range(iN):
        full_data[i*iT:(i+1)*iT,0] = np.unique(vId)[i]
        full_data[i*iT:(i+1)*iT,1] = np.unique(vT)

    #fill dataframe with available data
    full_data = pd.DataFrame(full_data).merge(pd.DataFrame(mX_random),on=[0,1],how='left')
    full_data.fillna(0,inplace=True)
    #capture X
    mX_random_final  = full_data[[3,4,5,6]]
    mX_random_final = sm.add_constant(mX_random_final)
    
    #capture y
    mY_random_final = full_data[2]
    
    #build OLS
    model_random = sm.OLS(mY_random_final,mX_random_final).fit(cov_type='HC1')
    
    #capture residuals
    resids = np.concatenate([np.array(full_data[0]).reshape(-1,1),np.array(model_random.resid).reshape(-1,1)],axis=1)
    
    #calculate average covariance matrix
    cov_group = np.zeros((iT,iT))
    for i in np.unique(vId):
        resid_group  = resids[resids[:,0]==i][:,1]
        cov_group  = cov_group + resid_group * resid_group.T
    
    cov_group = cov_group/iN
    
    
    mX_with_ids = np.concatenate([np.array(full_data[0]).reshape(-1,1),mX_random_final],axis=1)
    mY_with_ids = np.concatenate([np.array(full_data[0]).reshape(-1,1),np.array(mY_random_final).reshape(-1,1)],axis=1)
    
    
    #estimate random effects betas
    calc1 = np.zeros((5,5))
    calc2 = np.zeros((5,1))

    for i in np.unique(vId):
        x = mX_with_ids[mX_with_ids[:,0]==i][:,1:]
        y = mY_with_ids[mY_with_ids[:,0]==i][:,1].reshape(-1,1)
        calc1 = calc1 + x.T @  cov_group.T @ x
        calc2 = calc2 + x.T @  cov_group.T @ y
        
    beta_re = np.linalg.inv(calc1/iN) @  (1/iN *calc2)
    print(beta_re[1:])
    return beta_re[1:]

###########################################################
### main()
def main():
    # Magic numbers
    sIn= "data/Invest1993.xlsx"
    sY= 'inva'
    asX= ['vala', 'debta', 'cfa', 'nyseamex']
    sId= 'cusip'
    sIC= 'ardsic'

    # Initialisation
    [vId, vT, vY, mX, mA]= InitData(sIn, sId, sIC, sY, asX)

    # Output
    iN= np.unique(vId).shape[0]
    iT= np.unique(vT).shape[0]
    iO= vId.shape[0]
    iA= mA.shape[1]

    print ("Reading data from ", sIn)
    print ("Average y: ", np.mean(vY), ",\nAverage x: \n", np.mean(mX, axis=0))
    print ("Number of companies: %i, Number of years: %i, number of obs: %i, number of industries: %i" % (iN, iT, iO, iA))
    
    
    ##pooled
    pooled_estimators, pooled_stderr = Pooled(mX,mA,vY)
    

    ##fixed
    fixed_estimators, fixed_stderr = Fixed(mX,mA,vY,vId,iO,iN)
    
    
    ###random effect
    random_estimators = Random(mX,vY,vT,vId,iN,iT)
    
    
    
###########################################################
### start main
if __name__ == "__main__":
    main()
