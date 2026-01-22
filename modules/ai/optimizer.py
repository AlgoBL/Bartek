
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
        Fitness function: Maximize Skewness and Return, Minimize Volatility and Kurtosis.
        Fitness = Mean - Lambda*Vol + Gamma*Skew - Delta*Kurt
        """
        # Normalize weights to sum to 1
        weights = np.array(solution)
        if np.sum(weights) == 0:
            return -9999
        weights = weights / np.sum(weights)
        
        # Portfolio Returns
        port_returns = np.dot(self.returns_data, weights)
        
        # Metrics
        mean_ret = np.mean(port_returns) * 252
        vol = np.std(port_returns) * np.sqrt(252)
        sk = skew(port_returns)
        kt = kurtosis(port_returns)
        
        # Hyperparameters (Barbell Preference)
        # We want high return, high skewness (right tail), low kurtosis (no left tail risk ideally, but barbell accepts some)
        # However, Barbell seeks "Good" Kurtosis (convexity). 
        # Let's focus on simple equation from research.
        
        # Fitness = Returns + 0.5 * Skewness - 1.0 * Volatility
        fitness = mean_ret + 2.0 * sk - 0.5 * vol - 0.1 * kt
        
        return fitness

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
