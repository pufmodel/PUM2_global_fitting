# March 2018--WRB
# Script to fit additive consecutive binding model

############################################################################
# Import modules
import scipy
import pickle
import numpy as np
import pandas as pd
import sys
import os
import lmfit
from lmfit import minimize, Parameters, Parameter, report_fit
import copy

############################################################################
# Define Functions
def linear_register_ddG(sequence, base_penalties):
    # Function to compute exp(-ddG/kT) for the linear bases bound to the protein at positions 1-9
    # Inputs:
    # sequence--the sequence that the ddG is being computed for
    # base_penalties--base penalties as exp(-ddG_basepenalty/kT) for each base at each position listed in the following order: (1A, 1C, 1U, 1G, ...)
    # Outputs:
    # Ka_rel--return the relative Ka (1/kdrel = kd,WT/kd,register = exp(-ddG/RT)) for the sequence
    #########################################################################

    # initializethe relative Ka to 1
    Ka_rel = 1
    # apply base penalties at each position
    for i in range(len(sequence)):
        if sequence[i] == 'A':
            Ka_rel = Ka_rel*base_penalties[4*i]
        if sequence[i] == 'C':
            Ka_rel = Ka_rel*base_penalties[4*i+1]
        if sequence[i] == 'U':
            Ka_rel = Ka_rel*base_penalties[4*i+2]
        if sequence[i] == 'G':
            Ka_rel = Ka_rel*base_penalties[4*i+3]
    return Ka_rel


def initialize_Objective_Function_PUF_Does_Flips(initial_mutation_penalties, Temperature, mutation_range_high, mutation_range_low):
    # Function to initialize the model parameters and package them in a Parameters object for lmfit. Note that parameters are initialized
    # and fit as exp(-ddGparam/kT) because there is no reason to have to make the additional computation to compute the partition function from
    # the ddG values during fitting.
    # Inputs:
    # initial_mutational_penalties--provided as ddG values in kcal/mol
    # inital_flip_params--provided as ddG values in kcal/mol
    # mutation_range_high--upperbounds on the amount a base penalty can change from the initial value in kcal/mol
    # mutation_range_low--lowerbounds on the amount a base penalty can change from the inital value in kcal/mol
    # Temperature--Temperature (in degrees Celsius) that the experimental ddGs were collected at
    # Outputs:
    # params--Parameters object for lmfit
    # param_names--Names of the parameters to be fit

    #########################################################################
    # Define the ddG conversion factor (-kT)*2.30258509299 (the 2.3025 factor converts from ln to log10)
    ddG_conversion_factor = -(Temperature+273.15)*0.0019872041*2.30258509299

    #########################################################################
    # Define fit parameters and intialize as relative Ka effects (Karel = kd,WT/kd,register = exp(-ddG/RT))--Using relative Ka's allows parameters for a register to be multiplied and then go straight into the sum of the partition function
    fitParameters = pd.DataFrame(index=['upperbound', 'initial', 'lowerbound'],
                                 columns=['oneA', 'oneC', 'oneT', 'oneG', 'twoA', 'twoC', 'twoT', 'twoG', 
                                          'threeA', 'threeC', 'threeT', 'threeG', 'fourA', 'fourC', 'fourT', 'fourG', 
                                          'fiveA', 'fiveC', 'fiveT', 'fiveG', 'sixA', 'sixC', 'sixT', 'sixG', 
                                          'sevenA', 'sevenC', 'sevenT', 'sevenG', 'eightA', 'eightC', 'eightT', 'eightG',
                                          'nineA', 'nineC', 'nineT', 'nineG'])
        
    #########################################################################
    # Initialize mutational penalties
    # Compute upper and lower bounds for positional penalties
    initial_mutation_penalties_low = 10**((initial_mutation_penalties-mutation_range_low)/(ddG_conversion_factor))
    initial_mutation_penalties_high = 10**((initial_mutation_penalties+mutation_range_high)/(ddG_conversion_factor))
    initial_mutation_penalties = 10**(initial_mutation_penalties/(ddG_conversion_factor))

    k=0
    for i in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']:
        for j in ['A', 'C', 'G', 'T']:
            fitParameters.loc[:, i+j] = [initial_mutation_penalties_low[k],initial_mutation_penalties[k],initial_mutation_penalties_high[k]]
            k = k+1

    #########################################################################
    # Define consensus bases
    consensus_bases = ['oneT', 'twoG', 'threeT', 'fourA', 'fiveT', 'sixA', 'sevenT', 'eightA', 'nineT']

    #########################################################################
    # Define the names of the fit parameters.
    param_names = fitParameters.columns.tolist()
    
    #########################################################################
    # Store initial fit parameters in Parameters object for fitting with lmfit.
    params = Parameters()
    for param in param_names:
        if param in consensus_bases:
            params.add(param, value=fitParameters.loc['initial', param],
                   min = fitParameters.loc['lowerbound', param],
                   max = fitParameters.loc['upperbound', param], vary = False)
        else:
            params.add(param, value=fitParameters.loc['initial', param],
                   min = fitParameters.loc['lowerbound', param],
                   max = fitParameters.loc['upperbound', param])
        
    return params, param_names


def Objective_Function_PUF_Does_Flips(params, passed_sequences, data=None):
    # Inputs:
    # params--lmfit Parameters object created with initialize_Objective_Function_PUF_Does_Flips
    # passed_sequences--list of sequences to be fit
    # data--ddG values corresponding to the passed sequences
    # Outputs:
    # This function will return the values of the partition functions for a list of sequences if no data is given or the residuals 
    # between the ensemble ddG and the ddG values in the data.
    
    #########################################################################
    # Get parameter values from param object
    parvals = params.valuesdict()
    base_penalties = [parvals['oneA'], parvals['oneC'], parvals['oneT'], parvals['oneG'], parvals['twoA'], parvals['twoC'], parvals['twoT'], 
                        parvals['twoG'], parvals['threeA'], parvals['threeC'], parvals['threeT'], parvals['threeG'], parvals['fourA'], parvals['fourC'], 
                        parvals['fourT'], parvals['fourG'], parvals['fiveA'], parvals['fiveC'], parvals['fiveT'], parvals['fiveG'], parvals['sixA'], parvals['sixC'], 
                        parvals['sixT'], parvals['sixG'], parvals['sevenA'], parvals['sevenC'], parvals['sevenT'], parvals['sevenG'], parvals['eightA'], 
                        parvals['eightC'], parvals['eightT'], parvals['eightG'], parvals['nineA'], parvals['nineC'], parvals['nineT'], parvals['nineG']]

    #########################################################################
    # Deifine a list to store all of the partition function values
    sequence_partition_function_values = []
    
    #########################################################################
    #Iterate through the list of passed sequences    
    for i in range(len(passed_sequences)):
        sequence = list(passed_sequences[i])
        
        # Initialize a list to store all of the partition function terms (one corresponding to each register for a specific sequence)
        # that will sum to give the fill partition function
        single_ddG_values = []

        # Add registers with no flips
        for i in range(len(sequence)-8):
            single_ddG_values.append(linear_register_ddG(sequence[i:i+9], base_penalties))

        # Can add other model features here if you want to include registers with bulged bases, etc.

        # Compute partition function from all possible registers and save
        partition_function = np.sum(single_ddG_values)
        sequence_partition_function_values.append(partition_function)

    # compute ddG from the partition function value
    Temperature = 25
    ddG_conversion_factor = -(Temperature+273.15)*0.0019872041*2.30258509299
    ddG_space_values = ddG_conversion_factor*np.log10(sequence_partition_function_values)
        
    if data is None:
        return ddG_space_values
    else:
        # return residual if data is provided
        return ddG_space_values - data


def fit_mutational_params(training_sequences, training_dG, testing_sequences, testing_dG, initial_mutation_penalties, mutation_range_high, mutation_range_low, save_prefix, Temperature):
    # This function fits the ensmble model for PUF protein binding with the training sequences and ddGs provided. The fit parameters are
    # saved as a csv file and plots comparing the ddG and parameter sensitivities are made.
    # Inputs:
    # training_sequences--list of sequences to use to fit the model
    # training_dG--ddG values corresponding to the training sequences
    # testing_sequences--list of sequences to test the model
    # testing_dG--ddG values corresponding to the testing sequences
    # initial_mutation_penalties--Initial values of the mutational penalties
    # mutation_range_high--how much the single mutant penalties are allowed to increase
    # mutation_range_low--how much the single mutant penalties are allowed to decrease
    # initial_flip_params--initial flip penalties
    # save_prefix--prefix of where and what name to save the data with
    # Temperature--temperature that the ddG values were collected at
    # param_sensitivity--change to False to not do parameter sensitivity analysis


    # define initial parameters in a parameter object for lmfit
    params, param_names = initialize_Objective_Function_PUF_Does_Flips(initial_mutation_penalties, initial_flip_params, Temperature, mutation_range_high, mutation_range_low)

    # define the objective function to be fit
    func = Objective_Function_PUF_Does_Flips

    # fit the model with lmfit using the BFGS or differential evolution algorithms
    results = minimize(func, params,
            args = (np.array(list(training_sequences)), ),
            kws={'data':np.array(list(training_dG))},
            method = 'differential evolution')

    # compute the conversion factor for a given temperature
    ddG_conversion_factor = -(Temperature+273.15)*0.0019872041*2.30258509299

    # parse the results of the fit
    actual_params_list = [] # create a list to store the final parameter values
    initial_params = [] # create a list containing the initial values
    actual_params = [] # create a list of lists for parameters at each position (could also just reshape actual_params variable)
    current_list = []
    i = 0 # initialize a counter
    for param in param_names:
        i = i+1
        current_list.append(ddG_conversion_factor*np.log10(results.params[param].value))
        actual_params_list.append(ddG_conversion_factor*np.log10(results.params[param].value))
        initial_params.append(ddG_conversion_factor*np.log10(params[param].value))
        if i == 4:
            actual_params.append(current_list)
            current_list = []
            i = 0
    # add zeros to complete the final current list and append
    current_list.append(0)
    current_list.append(0)
    actual_params.append(current_list)

    # Convert the final parameters to a data frame and save
    pd.DataFrame(actual_params).to_csv(save_prefix + 'parameter_values.csv')

    return param_names, results.params


############################################################################
# main--fill in user defined inputs below
def main():
    Temperature = 25 # there is one other hardcoded temperature value in the objective function, so if you change this be sure to find and replace all temperature values!!!!! Note that in principle this is arbitrary since ddG values are provided

    # load the data--should include the sequence and the ddG value for each sequence
    dGvalues_training = pd.read_csv('./dGvalues_training.csv', sep = '\t')
    dGvalues_testing = pd.read_csv('./dGvalues_testing.csv', sep = '\t')
    
    # Create arrays contianing only the sequences and ddG values
    training_sequences = np.array(list(dGvalues_training['seq']))
    training_dG = np.array(list(dGvalues_training['ddG']))
    testing_sequences = np.array(list(dGvalues_testing['seq']))
    testing_dG = np.array(list(dGvalues_testing['ddG']))

    # define the initial single mutant penalties. These are set to the experimentally determined single mutant penalties here
    initial_params = np.array([2.848, 3.111, 2.642, 0,
                    1.722, 3.534, 0, 2.875,
                    2.448, 2.75, 2.408, 0,
                    0, 2.013, 0.868, 1.635,
                    0.080, 0.051, 0.814, 0,
                    0.00, 1.964, 1.836, 1.886,
                    1.48, 1.643, 1.412, 0,
                    0, 1.506, 1.123, 0.936,
                    0.082, 0.55, -0.34, 0.00])

    # define the amount the single mutant penalties are allowed to increase and decrease
    mutation_range_low = np.array([0.4,0.694,0.4,0.4,
                            0.4,0.4,0.4,0.4,
                            0.538,0.4,0.627,0.4,
                            0.4,0.4,0.4,0.4,
                            0.4,0.4,0.4,0.4,
                            0.4,0.4,0.755,0.4,
                            0.407,0.952,1.09,0.4,
                            0.4,0.4,0.4,0.4,
                            0.4,0.4,0.4,0.4])
    mutation_range_high = np.array([0.4,0.4,0.4,0.4,
                            0.4,0.4,0.4,0.4,
                            0.516,0.4,0.507,0.4,
                            0.4,0.4,0.846,0.4,
                            0.4,0.4,0.4,0.4,
                            0.4,0.4,0.4,0.4,
                            0.4,0.444,0.4,0.4,
                            0.4,0.4,0.4,0.4,
                            0.4,0.4,0.4,0.4])

    # define the save prefix
    savePrefix = './global_fit_linear_ensemble_model_'

    # perform the fit
    fit_mutational_params(training_sequences, training_dG, testing_sequences, testing_dG, initial_params, mutation_range_high, mutation_range_low, savePrefix, Temperature)

main()

