
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import streamlit as st

class PortfolioArchitect:
    def __init__(self):
        """
        Initializes the Portfolio Architect using Hierarchical Risk Parity (HRP).
        Manual implementation using SciPy to avoid PyPortfolioOpt/CVXPY dependency.
        """
        pass
        
    def getIVP(self, cov):
        """Compute the inverse-variance portfolio"""
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def getClusterVar(self, cov, cItems):
        """Compute variance per cluster"""
        cov_slice = cov.loc[cItems, cItems] # matrix slice
        w = self.getIVP(cov_slice).reshape(-1, 1)
        cVar = np.dot(np.dot(w.T, cov_slice), w)[0, 0]
        return cVar

    def getQuasiDiag(self, link):
        """Sort clustered items by distance"""
        link = link.astype(int)
        sortIx = pd.Series([link[-1, 0], link[-1, 1]])
        numItems = link[-1, 3] # number of original items
        
        while sortIx.max() >= numItems:
            sortIx.index = range(0, sortIx.shape[0] * 2, 2) # make space
            df0 = sortIx[sortIx >= numItems] # find clusters
            i = df0.index
            j = df0.values - numItems
            sortIx[i] = link[j, 0] # item 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sortIx = pd.concat([sortIx, df0]) # reorder
            sortIx = sortIx.sort_index()
            sortIx.index = range(sortIx.shape[0])
            
        return sortIx.tolist()

    def getRecBipart(self, cov, sortIx):
        """Compute HRP alloc"""
        w = pd.Series(1, index=sortIx)
        cItems = [sortIx] # initialize all items in one cluster
        
        while len(cItems) > 0:
            cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(cItems), 2):
                cItems0 = cItems[i] # cluster 1
                cItems1 = cItems[i + 1] # cluster 2
                
                cVar0 = self.getClusterVar(cov, pd.Index(cItems0))
                cVar1 = self.getClusterVar(cov, pd.Index(cItems1))
                
                alpha = 1 - cVar0 / (cVar0 + cVar1)
                w[cItems0] *= alpha # weight 1
                w[cItems1] *= 1 - alpha # weight 2
        return w

    def allocate_hrp(self, prices_df):
        """
        Calculates HRP weights for the given assets.
        """
        try:
            # Calculate returns and covariance
            returns = prices_df.pct_change().dropna()
            cov = returns.cov()
            corr = returns.corr()
            
            # 1. Clustering
            dist = np.sqrt((1 - corr) / 2)
            link = sch.linkage(dist, 'single')
            
            # 2. Sort
            sortIx = self.getQuasiDiag(link)
            sortIx = corr.index[sortIx].tolist()
            
            # 3. Allocation
            hrp_weights = self.getRecBipart(cov, sortIx)
            
            return hrp_weights.to_dict()
            
        except Exception as e:
            # st.error(f"HRP Allocation Error: {e}")
            # Fallback to Equal Weight
            n = len(prices_df.columns)
            return {ticker: 1.0/n for ticker in prices_df.columns}
