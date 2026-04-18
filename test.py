import numpy as np

# The numerical computation of the Hessian may yield
# some warnings. These can be ignored using:
import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fmin_slsqp
from scipy.special import gammaln
from scipy.stats import multivariate_normal
import math

import timeit




# For computation of the numerical Hessian (for the standard errors):
# (I had to first install the numdifftools module 
# with the line "pip install numdifftools" in Anaconda prompt.)
import numdifftools as nd

     
# For inverting the Hessian matrix, for computing -Hessian^{-1}:
from numpy.linalg import inv


# Negative Log-Likelihood Function (to be minimized):
def fMinusLogLikelihoodGARCHStudentT(vTheta): 
    
    dOmega = vTheta[0]
    dAlpha = vTheta[1]
    dBeta  = vTheta[2]
    dNu    = vTheta[3]
    
    # Compute the variances vH[t] in the GARCH model 
    # (starting with the sample variance):
    
    iT = len(g_vR)  
    vH = np.ones((iT,1)) 
    
    vH[0] = np.var(g_vR)
    
    for t in range(1,iT):
        vH[t] = dOmega + dAlpha*g_vR[t-1]**2 + dBeta*vH[t-1]
    
    # Compute vector of log(density) values:
    # Note: \ symbol for code that continues on multiple lines:
    
    vLogPdf = gammaln( (dNu+1)/2 ) - gammaln( dNu/2 )                \
              - 0.5*np.log( (dNu-2)*np.pi*vH )                       \
              - 0.5*(dNu+1)*np.log( 1 +  (g_vR**2)/((dNu-2)*vH) )      
              
    dMinusLogLikelihood = - np.sum(vLogPdf)
    
    return dMinusLogLikelihood


   

def main():
   
     # Global variable g_vR: data of the log-returns:
    
    global g_vR


    # Load data and compute log-returns:
  
    sIn = 'SP500_1998-2007.csv'
    df  = pd.read_csv(sIn, index_col = 'Date',parse_dates=True)
    df.drop(['Open', 'High', 'Low', 'Close', 'Volume'],axis=1,inplace=True)
    vPrice = df.values;
    iNumPrices = len(vPrice);
    vReturns = 100 * ( np.log(vPrice[1:iNumPrices]) - np.log(vPrice[0:(iNumPrices-1)]) );
    
    # Select estimation window of in-sample observations:
    
    iNumInSampleObservations = 1500;
    g_vR = vReturns[0:iNumInSampleObservations];
    
    
    # Initial values for optimization:
    
    dOmega_ini = 0.1
    dAlpha_ini = 0.05
    dBeta_ini  = 0.90
    dNu_ini    = 6    
        
    vTheta_ini = [dOmega_ini, dAlpha_ini, dBeta_ini, dNu_ini]
    

    # Restrictions for optimization: 
    vOmega_bnds   = (0, 1)
    vAlpha_bnds   = (0, 1)
    vBeta_bnds    = (0, 1)
    vNu_bnds      = (2.1, 30)

    vTheta_bnds = [vOmega_bnds, vAlpha_bnds, vBeta_bnds, vNu_bnds]

    
    # Estimation: minimize fMinusLogLikeStudentT function using fmin_slsqp command:
    
    vTheta_ML = fmin_slsqp(fMinusLogLikelihoodGARCHStudentT, vTheta_ini, bounds = vTheta_bnds)
                       
       
    # Compute standard errors as sqrt of diagonal elements
    # of estimated covariance matrix of ML estimator, 
    # where estimated covariance matrix = -(Hessian of loglikelihood evaluated at MLE)^{-1}.
    # Note: the function already gives -loglikelihood, here we just need
    #       Hessian^{-1} instead of -Hessian^{-1}.
    
    mHessianofMinusLogL = nd.Hessian(fMinusLogLikelihoodGARCHStudentT)(vTheta_ML)
    mCovariance_MLE     = inv(mHessianofMinusLogL)
    vStandardErrors     = np.sqrt(np.diag(mCovariance_MLE)) 


    # Print results with 4 decimals:
    
    np.set_printoptions(precision=4)
    
    print("loglikelihood: ", -fMinusLogLikelihoodGARCHStudentT(vTheta_ML) );
    
    print( "ML estimator and standard errors: ");
    print( np.stack( (vTheta_ML, vStandardErrors), axis=1 ) );
    
    
    print("ML estimator: omega: ", "%.4f" % vTheta_ML[0], " (" , "%.4f" % vStandardErrors[0] ,  ")")
    print("ML estimator: alpha: ", "%.4f" % vTheta_ML[1], " (" , "%.4f" % vStandardErrors[1] ,  ")")
    print("ML estimator: beta:  ", "%.4f" % vTheta_ML[2], " (" , "%.4f" % vStandardErrors[2] ,  ")")
    print("ML estimator: nu:    ", "%.4f" % vTheta_ML[3], " (" , "%.4f" % vStandardErrors[3] ,  ")")
          
    
    
    
    
    # Random walk Metropolis-Hastings method:
    
    print("Random walk Metropolis-Hastings method:")
   
    dTimeStart = timeit.default_timer()  # for keeping track of computing time


    iNdraws = 1100            # total number of draws
    iBurnIn =  100            # number of burn-in draws
    iNumAcceptance = 0
    
    # matrix for stpring the accepted (and repeated) draws:
    mTheta = np.zeros((iNdraws,len(vTheta_ML)))   
    
    # feasible initial value: theta_0 = the ML estimator:
    mTheta[0] = vTheta_ML
    
    for i in range(1,iNdraws):
        
       # candidate draw: theta~ ~ N(theta_{i-1}, Sigma) with Sigma = cov(theta_ML):
       vTheta_candidate = multivariate_normal.rvs( mTheta[i-1] , mCovariance_MLE )
       
       # if outside the allowed domain: reject immediately:
       
       if (vTheta_candidate[0] < 0) or (vTheta_candidate[1] < 0) or (vTheta_candidate[2] < 0) or (vTheta_candidate[3] < 2) or (vTheta_candidate[3] > 30):
           mTheta[i] = mTheta[i-1]
       else:
       # if inside the allowed domain: compute acceptance probability and accept/reject candidate draw:  
       # Note: fMinusLogLikelihoodGARCHStudentT gives -loglikelihood:
           dAccProb         = min( np.exp( -fMinusLogLikelihoodGARCHStudentT(vTheta_candidate) + fMinusLogLikelihoodGARCHStudentT(mTheta[i-1]) ), 1 )
           dU               = np.random.uniform(0,1) 
           if dU < dAccProb:
               mTheta[i] = vTheta_candidate
               iNumAcceptance = iNumAcceptance+1
           else:
               mTheta[i] = mTheta[i-1]
                   
        # After each 100 steps report the computing time:           
       if (np.mod(i+1,100)==0):
           dTime = timeit.default_timer()
           print(i+1, " draws in ", dTime - dTimeStart, " seconds") 
                        
           
    # Compute acceptance percentage:
    
    dAcceptancePercentage = 100*iNumAcceptance/iNdraws
    print( "acceptance percentage: ", dAcceptancePercentage )
    
    # Estimate posterior mean & stdev as sample mean & sample stdev
    # of draws after the burn-in:
    
    vTheta_posterior_mean  = np.mean(mTheta[iBurnIn:], axis = 0)
    vTheta_posterior_stdev = np.std( mTheta[iBurnIn:], axis = 0)
    
    print("posterior means:", vTheta_posterior_mean)
    print("posterior stdev:", vTheta_posterior_stdev)
    
    
    # Histograms of draws from posterior distribution (draws after burn-in):
    
    plt.hist(mTheta[iBurnIn:,0], bins = 20)
    plt.show()
    plt.hist(mTheta[iBurnIn:,1], bins = 20)
    plt.show()
    plt.hist(mTheta[iBurnIn:,2], bins = 20)
    plt.show()
    plt.hist(mTheta[iBurnIn:,3], bins = 20)
    plt.show()

    
    
    
if __name__ == "__main__":
    main()