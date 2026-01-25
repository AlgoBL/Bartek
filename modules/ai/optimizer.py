
import numpy as np
import pandas as pd
import pygad
import streamlit as st
from scipy.stats import skew, kurtosis

class GeneticOptimizer:
    def __init__(self, num_generations=50):
        self.num_generations = num_generations
        self.sol_per_pop = 20
        self.num_parents_mating = 10
        self.returns_data = None
        self.risk_free_rate = 0.05
    
    def fitness_func(self, ga_instance, solution, solution_idx):
        """
        Fitness function: Maximize Sortino Ratio.
        Sortino = (Mean_Return - Threshold) / Downside_Deviation
        """
        # Normalize weights to sum to 1
        weights = np.array(solution)
        if np.sum(weights) == 0:
            return -9999
        weights = weights / np.sum(weights)
        
        # Portfolio Returns
        port_returns = np.dot(self.returns_data, weights)
        
        # Parameters
        rf = self.risk_free_rate / 252 # Daily RF
        target_return = 0 # Can be set to RF or 0
        
        # Average Daily Return
        mean_daily_ret = np.mean(port_returns)
        
        # Downside Deviation calculation
        # Filter only returns below target
        negative_returns = port_returns[port_returns < target_return]
        
        if len(negative_returns) == 0:
            downside_std = 0.000001 # Avoid division by zero if no losses
        else:
            downside_std = np.std(negative_returns)
            
        # Annualized Measures
        # Note: Sortino scaling is debated, but commonly sqrt(252) applied to the ratio or individually
        annualized_return = mean_daily_ret * 252
        annualized_downside_std = downside_std * np.sqrt(252)
        
        if annualized_downside_std == 0:
            sortino = 9999 # Excellent portfolio
        else:
            sortino = (annualized_return - self.risk_free_rate) / annualized_downside_std
            
        # Add a penalty for extreme kurtosis if desired, or just trust Sortino
        # Barbell strategy loves 'good' convexity, so we stick to Sortino which allows upside vol.
        
        return sortino

    def optimize_portfolio(self, returns_df):
        """
        Runs the Genetic Algorithm to find optimal weights.
        """
        self.returns_data = returns_df.values
        num_genes = returns_df.shape[1]
        
        # GA Parameters
        ga_instance = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            fitness_func=self.fitness_func,
            sol_per_pop=self.sol_per_pop,
            num_genes=num_genes,
            init_range_low=0.0,
            init_range_high=1.0,
            gene_type=float,
            mutation_percent_genes=20
        )
        
        ga_instance.run()
        
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        
        # Normalize best solution weights
        best_weights = solution / np.sum(solution)
        
        weight_dict = {col: w for col, w in zip(returns_df.columns, best_weights)}
        return weight_dict
