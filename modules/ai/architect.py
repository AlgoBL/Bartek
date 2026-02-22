
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

    def getCVaR(self, returns, alpha=0.05):
        """Compute Conditional Value at Risk (Expected Shortfall) at alpha confidence level"""
        sorted_returns = np.sort(returns)
        index = int(alpha * len(sorted_returns))
        cvar = np.abs(sorted_returns[:index].mean())
        return cvar

    def getClusterRisk(self, returns, cItems, metric='variance'):
        """Compute risk per cluster (Variance or CVaR)"""
        if metric == 'cvar':
            # Calculate portfolio returns for uniformity assuming equal weights within cluster for risk est
            # A better approach for HRP is often to treat the cluster as a single unit.
            # Here we estimate risk of the cluster formed by equal weighting
            cluster_returns = returns[cItems].mean(axis=1)
            return self.getCVaR(cluster_returns.values)
        else:
            # Standard Variance
            cov = returns.cov()
            cov_slice = cov.loc[cItems, cItems]
            w = self.getIVP(cov_slice).reshape(-1, 1)
            return np.dot(np.dot(w.T, cov_slice), w)[0, 0]

    def getRecBipart(self, returns, sortIx, metric='variance'):
        """Compute HRP alloc using specific risk metric"""
        w = pd.Series(1.0, index=sortIx)
        cItems = [sortIx] # initialize all items in one cluster
        
        while len(cItems) > 0:
            cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(cItems), 2):
                cItems0 = cItems[i] # cluster 1
                cItems1 = cItems[i + 1] # cluster 2
                
                cVar0 = self.getClusterRisk(returns, pd.Index(cItems0), metric=metric)
                cVar1 = self.getClusterRisk(returns, pd.Index(cItems1), metric=metric)
                
                # Protect against zero division
                if cVar0 + cVar1 == 0:
                    alpha = 0.5
                else:
                    alpha = 1 - cVar0 / (cVar0 + cVar1)
                
                w[cItems0] *= alpha # weight 1
                w[cItems1] *= 1 - alpha # weight 2
        return w

    def getQuasiDiag(self, link):
        """
        Sort clustered items by distance
        """
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
            sortIx = pd.concat([sortIx, df0]) # item 2
            sortIx = sortIx.sort_index() # re-sort
            sortIx.index = range(sortIx.shape[0]) # re-index
            
        return sortIx.tolist()

    def allocate_hrp(self, prices_df, risk_metric='cvar'):
        """
        Calculates HRP weights for the given assets.
        risk_metric: 'variance' or 'cvar' (Conditional Value at Risk / Expected Shortfall)
        """
        try:
            # Calculate returns and covariance
            returns = prices_df.pct_change().dropna()
            # cov = returns.cov() # Not strictly needed for allocation now, but needed for clustering
            corr = returns.corr()
            
            # 1. Clustering
            dist = np.sqrt((1 - corr) / 2)
            link = sch.linkage(dist, 'single')
            
            # 2. Sort
            sortIx = self.getQuasiDiag(link)
            sortIx = corr.index[sortIx].tolist()
            
            # 3. Allocation
            # Pass full returns df because CVaR needs historical distribution, not just covariance
            hrp_weights = self.getRecBipart(returns, sortIx, metric=risk_metric)
            
            return hrp_weights.to_dict()
            
        except Exception as e:
            print(f"DEBUG ERROR: {e}")
            import traceback
            traceback.print_exc()
            # st.error(f"HRP Allocation Error: {e}")
            # Fallback to Equal Weight
            n = len(prices_df.columns)
            return {ticker: 1.0/n for ticker in prices_df.columns}

    def allocate_cvar_risk_parity(self, prices_df: pd.DataFrame, alpha: float = 0.05) -> dict:
        """
        CVaR Risk Parity allocation.
        Equalizes each asset's Conditional Value at Risk (Expected Shortfall) contribution.
        Reference: Rockafellar & Uryasev (2000), Boudt et al. (2012).

        Each asset's CVaR contribution should equal 1/n of portfolio CVaR.
        Implemented as iterative proportional scaling (no CVXPY needed).
        """
        returns = prices_df.pct_change().dropna()
        n_assets = len(returns.columns)
        tickers = returns.columns.tolist()

        if n_assets < 2:
            return {t: 1.0 for t in tickers}

        # Start from equal weights
        weights = np.ones(n_assets) / n_assets

        for iteration in range(200):
            # Portfolio returns with current weights
            port_returns = returns.values @ weights

            # Portfolio CVaR at alpha level
            cutoff = np.percentile(port_returns, alpha * 100)
            tail_returns = port_returns[port_returns <= cutoff]
            port_cvar = float(np.abs(tail_returns.mean())) if len(tail_returns) > 0 else 1e-6

            # Marginal CVaR for each asset (numerical differentiation)
            marginal_cvar = np.zeros(n_assets)
            eps = 1e-4
            for i in range(n_assets):
                w_plus = weights.copy()
                w_plus[i] += eps
                w_plus /= w_plus.sum()
                port_plus = returns.values @ w_plus
                cutoff_plus = np.percentile(port_plus, alpha * 100)
                tail_plus = port_plus[port_plus <= cutoff_plus]
                cvar_plus = float(np.abs(tail_plus.mean())) if len(tail_plus) > 0 else 1e-6
                marginal_cvar[i] = (cvar_plus - port_cvar) / eps

            # Component CVaR = weight * marginal CVaR
            component_cvar = weights * np.abs(marginal_cvar)
            component_cvar = np.maximum(component_cvar, 1e-8)  # avoid zero

            # Target: equal CVaR contribution
            target_cvar = component_cvar.sum() / n_assets

            # Scale weights inversely proportional to their CVaR contribution
            new_weights = weights * (target_cvar / component_cvar)
            new_weights = np.maximum(new_weights, 0)  # long only
            new_weights /= new_weights.sum()

            # Check convergence
            if np.max(np.abs(new_weights - weights)) < 1e-6:
                break
            weights = new_weights

        return {t: float(w) for t, w in zip(tickers, weights)}

