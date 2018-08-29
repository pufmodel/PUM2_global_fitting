# March 2018--WRB
# Script to fit ensemble model with flips and coupling terms for PUM2 binding affinity

############################################################################
# Import modules
import scipy
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import sys
import os
import lmfit
from lmfit import minimize, Parameters, Parameter, report_fit
import seaborn as sns
import copy
sns.set_style('ticks')
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

############################################################################
# Define Functions
def linear_register_ddG_coupling_terms(sequence, base_penalties, coupling_terms, first_coupling, second_coupling):
    # Function to compute exp(-ddG/kT) for the nonflipped bases bound to the protein and position 9
    # Inputs:
    # sequence--the sequence that the ddG is being computed for
    # base_penalties--base penalties as relative Kds for each base at each position exp(-ddG_basepenalty/kT) listed in the following order: (1A, 1C, 1U, 1G, ...)
    # coupling_terms--penalties as relative Kds for the two coupling terms (567G8 and 67C8)
    # first_coupling--boolean value: if True include the first coupling term in the ddG calculation
    # second_coupling--boolean value: if True include the second coupling term in the ddG calculation
    # Outputs:
    # Ka_rel--return the relative Ka (1/kdrel = kd,WT/kd,register = exp(-ddG/RT)) for the sequence
    #########################################################################

    # initialize the relative Ka to 1
    Ka_rel = 1
    # apply base penalties at each position
    for i in range(len(sequence)):
        if i != 8 or sequence[7] == 'A':
            if sequence[i] == 'A':
                Ka_rel = Ka_rel*base_penalties[4*i]
            if sequence[i] == 'C':
                Ka_rel = Ka_rel*base_penalties[4*i+1]
            if sequence[i] == 'U':
                Ka_rel = Ka_rel*base_penalties[4*i+2]
            if sequence[i] == 'G':
                Ka_rel = Ka_rel*base_penalties[4*i+3]
    # include coupling terms if conditions are met and they were specified to be included
    if first_coupling:
        if sequence[4] == 'U' or sequence[4] == 'C':
            if sequence[5] == 'A':
                if sequence[6] == 'G':
                    if sequence[7] != 'A':
                        Ka_rel = Ka_rel*coupling_terms[0]
    if second_coupling:
        if sequence[5] != 'A':
            if sequence[6] == 'C':
                if sequence[7] != 'A':
                    Ka_rel = Ka_rel*coupling_terms[1]
    return Ka_rel


def initialize_Objective_Function_PUF_Does_Flips(initial_mutation_penalties, initial_flip_params, Temperature, mutation_range_high, mutation_range_low):
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
                                 columns=['three_A_flip', 'three_C_flip', 'three_T_flip', 'three_G_flip', 
                                          'four_A_flip', 'four_C_flip', 'four_T_flip', 'four_G_flip', 
                                          'five_A_flip', 'five_C_flip', 'five_T_flip', 'five_G_flip', 
                                          'six_A_flip', 'six_C_flip', 'six_T_flip', 'six_G_flip', 
                                          'oneA', 'oneC', 'oneT', 'oneG', 'twoA', 'twoC', 'twoT', 'twoG', 
                                          'threeA', 'threeC', 'threeT', 'threeG', 'fourA', 'fourC', 'fourT', 'fourG', 
                                          'fiveA', 'fiveC', 'fiveT', 'fiveG', 'sixA', 'sixC', 'sixT', 'sixG', 
                                          'sevenA', 'sevenC', 'sevenT', 'sevenG', 'eightA', 'eightC', 'eightT', 'eightG',
                                          'nineA', 'nineC', 'nineT', 'nineG', 'three_double_flip', 'four_double_flip', 
                                          'five_double_flip', 'six_double_flip', 'couple_567G8', 'couple_7C8'])
    
    #########################################################################
    # Initialize Double flips (min penalty: 0 kcal/mol; max penalty: 7 kcal/mol; initial penalty is double the average initial flip penalty at that position)
    fitParameters.loc[:, 'three_double_flip'] = [10**(0/ddG_conversion_factor),10**(np.mean([initial_flip_params[1], initial_flip_params[2], initial_flip_params[3]])*2/ddG_conversion_factor),10**(7/ddG_conversion_factor)]
    fitParameters.loc[:, 'four_double_flip'] = [10**(0/ddG_conversion_factor),10**(np.mean([initial_flip_params[4], initial_flip_params[6], initial_flip_params[7]])*2/ddG_conversion_factor),10**(7/ddG_conversion_factor)]
    fitParameters.loc[:, 'five_double_flip'] = [10**(0/ddG_conversion_factor),10**(np.mean([initial_flip_params[9], initial_flip_params[10], initial_flip_params[11]])*2/ddG_conversion_factor),10**(7/ddG_conversion_factor)]
    fitParameters.loc[:, 'six_double_flip'] = [10**(0/ddG_conversion_factor),10**(np.mean([initial_flip_params[13], initial_flip_params[14], initial_flip_params[15]])*2/ddG_conversion_factor),10**(7/ddG_conversion_factor)]

    #########################################################################
    # Initialize single flips (min penalty: 0 kcal/mol; max penalty: 7 kcal/mol)
    k=0
    for i in ['three', 'four', 'five', 'six']:
        for j in ['A', 'C', 'T', 'G']:
            fitParameters.loc[:, i+'_'+j+'_flip'] = [10**(0/ddG_conversion_factor),10**(initial_flip_params[k]/ddG_conversion_factor),10**(7/ddG_conversion_factor)]
            k = k+1

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
    # Initialize the coupling parameters
    fitParameters.loc[:, 'couple_567G8'] = [10**(-4/ddG_conversion_factor), 10**(0/ddG_conversion_factor), 10**(0/ddG_conversion_factor)]
    fitParameters.loc[:, 'couple_7C8'] = [10**(-4/ddG_conversion_factor), 10**(0/ddG_conversion_factor), 10**(0/ddG_conversion_factor)]
    
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


def get_flip_penalty(base, A_penalty, C_penalty, G_penalty, T_penalty):
    # Function to return the correct base penalty for a given base and set of penalties
    # Inputs:
    # base--actual base identity
    # A_penalty, C_penalty, G_penalty, and T_penalty--penalties for the identity of the base
    # Outputs:
    # Ka_rel--correct penalty for the base
    if base == 'A':
        Ka_rel = A_penalty
    if base == 'C':
        Ka_rel = C_penalty
    if base == 'G':
        Ka_rel = G_penalty
    if base == 'U':
        Ka_rel = T_penalty
    return Ka_rel


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
    three_A_flip = parvals['three_A_flip']
    three_C_flip = parvals['three_C_flip']
    three_T_flip = parvals['three_T_flip']
    three_G_flip = parvals['three_G_flip']
    four_A_flip = parvals['four_A_flip']
    four_C_flip = parvals['four_C_flip']
    four_T_flip = parvals['four_T_flip']
    four_G_flip = parvals['four_G_flip']
    five_A_flip = parvals['five_A_flip']
    five_C_flip = parvals['five_C_flip']
    five_T_flip = parvals['five_T_flip']
    five_G_flip = parvals['five_G_flip']
    six_A_flip = parvals['six_A_flip']
    six_C_flip = parvals['six_C_flip']
    six_T_flip = parvals['six_T_flip']
    six_G_flip = parvals['six_G_flip']
    three_double_flip = parvals['three_double_flip']
    four_double_flip = parvals['four_double_flip']
    five_double_flip = parvals['five_double_flip']
    six_double_flip = parvals['six_double_flip']
    base_penalties = [parvals['oneA'], parvals['oneC'], parvals['oneT'], parvals['oneG'], parvals['twoA'], parvals['twoC'], parvals['twoT'], 
                        parvals['twoG'], parvals['threeA'], parvals['threeC'], parvals['threeT'], parvals['threeG'], parvals['fourA'], parvals['fourC'], 
                        parvals['fourT'], parvals['fourG'], parvals['fiveA'], parvals['fiveC'], parvals['fiveT'], parvals['fiveG'], parvals['sixA'], parvals['sixC'], 
                        parvals['sixT'], parvals['sixG'], parvals['sevenA'], parvals['sevenC'], parvals['sevenT'], parvals['sevenG'], parvals['eightA'], 
                        parvals['eightC'], parvals['eightT'], parvals['eightG'], parvals['nineA'], parvals['nineC'], parvals['nineT'], parvals['nineG']]
    coupling_terms = [parvals['couple_567G8'], parvals['couple_7C8']]

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
            single_ddG_values.append(linear_register_ddG_coupling_terms(sequence[i:i+9], base_penalties, coupling_terms, True, True))

        # Add registers with single flips--only include coupling terms if not interupted by flips
        for i in range(len(sequence)-9):
            current_sequence = sequence[i:i+10]
            for j in range(2,6):
                if j == 2:
                    dG = linear_register_ddG_coupling_terms(current_sequence[0:j+1]+current_sequence[j+2::], base_penalties, coupling_terms, True, True)
                    single_ddG_values.append(dG*get_flip_penalty(current_sequence[j+1], three_A_flip, three_C_flip, three_G_flip, three_T_flip))
                if j == 3:
                    dG = linear_register_ddG_coupling_terms(current_sequence[0:j+1]+current_sequence[j+2::], base_penalties, coupling_terms, True, True)
                    single_ddG_values.append(dG*get_flip_penalty(current_sequence[j+1], four_A_flip, four_C_flip, four_G_flip, four_T_flip))
                if j == 4:
                    dG = linear_register_ddG_coupling_terms(current_sequence[0:j+1]+current_sequence[j+2::], base_penalties, coupling_terms, False, True)
                    single_ddG_values.append(dG*get_flip_penalty(current_sequence[j+1], five_A_flip, five_C_flip, five_G_flip, five_T_flip))
                if j == 5:
                    dG = linear_register_ddG_coupling_terms(current_sequence[0:j+1]+current_sequence[j+2::], base_penalties, coupling_terms, False, False)
                    single_ddG_values.append(dG*get_flip_penalty(current_sequence[j+1], six_A_flip, six_C_flip, six_G_flip, six_T_flip))
        
        # Add registers with 2x1nt flips--only include coupling terms if not interupted by flips
        for i in range(len(sequence)-10):
            current_sequence = sequence[i:i+11]
            for j in range(2,6):
                for k in range(j+1,6):
                    if j == 2:
                        dG = get_flip_penalty(current_sequence[j+1], three_A_flip, three_C_flip, three_G_flip, three_T_flip)
                    if j == 3:
                        dG = get_flip_penalty(current_sequence[j+1], four_A_flip, four_C_flip, four_G_flip, four_T_flip)
                    if j == 4:
                        dG = get_flip_penalty(current_sequence[j+1], five_A_flip, five_C_flip, five_G_flip, five_T_flip)
                    if j == 5:
                        dG = get_flip_penalty(current_sequence[j+1], six_A_flip, six_C_flip, six_G_flip, six_T_flip)
                    if k == 3:
                        dG = dG*linear_register_ddG_coupling_terms(current_sequence[0:j+1]+current_sequence[j+2:k+2]+current_sequence[k+3::], base_penalties, coupling_terms, True, True)
                        single_ddG_values.append(dG*get_flip_penalty(current_sequence[k+2], four_A_flip, four_C_flip, four_G_flip, four_T_flip))
                    if k == 4:
                        dG = dG*linear_register_ddG_coupling_terms(current_sequence[0:j+1]+current_sequence[j+2:k+2]+current_sequence[k+3::], base_penalties, coupling_terms, False, True)
                        single_ddG_values.append(dG*get_flip_penalty(current_sequence[k+2], five_A_flip, five_C_flip, five_G_flip, five_T_flip))
                    if k == 5:
                        dG = dG*linear_register_ddG_coupling_terms(current_sequence[0:j+1]+current_sequence[j+2:k+2]+current_sequence[k+3::], base_penalties, coupling_terms, False, False)
                        single_ddG_values.append(dG*get_flip_penalty(current_sequence[k+2], six_A_flip, six_C_flip, six_G_flip, six_T_flip))
        
        # Add registers with double flips
        for i in range(len(sequence)-10):
            current_sequence = sequence[i:i+11]
            for j in range(2,6):
                if j == 2:
                    dG = linear_register_ddG_coupling_terms(current_sequence[0:j+1]+current_sequence[j+3::], base_penalties, coupling_terms, True, True)
                    single_ddG_values.append(three_double_flip*dG)
                if j == 3:
                    dG = linear_register_ddG_coupling_terms(current_sequence[0:j+1]+current_sequence[j+3::], base_penalties, coupling_terms, True, True)
                    single_ddG_values.append(four_double_flip*dG)
                if j == 4:
                    dG = linear_register_ddG_coupling_terms(current_sequence[0:j+1]+current_sequence[j+3::], base_penalties, coupling_terms, False, True)
                    single_ddG_values.append(five_double_flip*dG)
                if j == 5:
                    dG = linear_register_ddG_coupling_terms(current_sequence[0:j+1]+current_sequence[j+3::], base_penalties, coupling_terms, False, False)
                    single_ddG_values.append(six_double_flip*dG)

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


def param_sensitivity_plots(results, func, params, param_names, Temperature, training_sequences, testing_sequences, save_prefix, training_dG, testing_dG, training_param_sensitivity = False):
    # Function to make plots of the sensitivity of the RMSE to each parameter
    # Inputs:
    # results--results object from lmfit
    # param_names--names of the parameters
    # func--objective function
    # params--Parameters object used in the fit
    # Temperature--Temperature ddG values were collected at
    # training_sequences--list of training sequences
    # testing_sequences--list of testing sequences
    # training_dG--list of ddG values for training sequences
    # testing_dG--list of ddG values for testing sequences
    # save_prefix--where to save the plots
    # training_param_sensitivity--True if you want to make parameter sensitivity plots for the traiing data as well (default False)

    training_RMSE = []
    testing_RMSE = []
    true_values = []
    all_param_values = []
    lower_bounds = []
    upper_bounds = []

    ddG_conversion_factor = -(Temperature+273.15)*0.0019872041*2.30258509299

    for param in param_names:
        newParams = copy.deepcopy(results.params)
        fit_value = results.params[param].value
        true_values.append(fit_value)
        lower_bounds.append([params[param].min])
        upper_bounds.append([params[param].max])

        current_train_RMSE = []
        current_test_RMSE = []
        param_values = []
        
        test_param_values = 10**(np.linspace(ddG_conversion_factor*np.log10(params[param].min), ddG_conversion_factor*np.log10(params[param].max), 25)/ddG_conversion_factor)
        for i in test_param_values:
            newParams[param].value = i
            fitdGsAll = func(newParams, np.array(list(testing_sequences)))
            RMSD_testing = np.sqrt(np.sum((np.array(fitdGsAll)-np.array(testing_dG))**2)/len(fitdGsAll))
            current_test_RMSE.append(RMSD_testing)
            param_values.append(i)
            if training_param_sensitivity:
                fitdGs = func(newParams, np.array(list(training_sequences)))
                RMSD_training = np.sqrt(np.sum((np.array(fitdGs)-np.array(training_dG))**2)/len(fitdGs))
                current_train_RMSE.append(RMSD_training)

        all_param_values.append(param_values)
        if training_param_sensitivity:
            training_RMSE.append(current_train_RMSE)
        testing_RMSE.append(current_test_RMSE)

    f, axarr = plt.subplots((len(all_param_values))/4+1,4, figsize=(15,len(all_param_values)+1))
    j = 0
    k = 0
    for i in range(len(all_param_values)):
        axarr[j,k].plot(ddG_conversion_factor*np.log10(all_param_values[i]), testing_RMSE[i], 'bo')
        axarr[j,k].plot(ddG_conversion_factor*np.log10([true_values[i], true_values[i]]), [min(testing_RMSE[i])*0.9, max(testing_RMSE[i])*1.1], 'k-')
        axarr[j,k].set_ylim(min(testing_RMSE[i])*0.9, max(testing_RMSE[i])*1.1)
        axarr[j,k].set_title(param_names[i])
        k = k+1
        if k == 4:
            k=0
            j = j+1

    f.text(0.04, 0.5, 'RMSE (kcal/mol)', va='center', rotation='vertical', fontsize = 40)
    f.text(0.5, 0.05, '$\Delta \Delta G$', ha='center', fontsize = 40)

    f.savefig(save_prefix + '_Parameter_Sensitivity_Testing.pdf', dpi=400)

    if training_param_sensitivity:
        f, axarr = plt.subplots((len(all_param_values))/4+1,4, figsize=(15,len(all_param_values)+1))
        j = 0
        k = 0
        for i in range(len(all_param_values)):
            axarr[j,k].plot(ddG_conversion_factor*np.log10(all_param_values[i]), training_RMSE[i], 'bo')
            axarr[j,k].plot(ddG_conversion_factor*np.log10([true_values[i], true_values[i]]), [min(training_RMSE[i])*0.9, max(training_RMSE[i])*1.1], 'k-')
            axarr[j,k].set_ylim(min(training_RMSE[i])*0.9, max(training_RMSE[i])*1.1)
            axarr[j,k].set_title(param_names[i])
            k = k+1
            if k == 4:
                k=0
                j = j+1

        f.text(0.04, 0.5, 'RMSE (kcal/mol)', va='center', rotation='vertical', fontsize = 40)
        f.text(0.5, 0.05, '$\Delta \Delta G$', ha='center', fontsize = 40)

        f.savefig(save_prefix + '_Parameter_Sensitivity_Training.pdf', dpi=400)


def ddG_comparison_plot(fitdGs, training_dG, save_name):
    # Make a plot of the ddG values computed by the fit model vs the experimental ddG values
    RMSD_fit = np.sqrt(np.sum((np.array(fitdGs)-np.array(training_dG))**2)/len(fitdGs))
    num_fit = len(fitdGs)

    fig, ax = plt.subplots(figsize=(9,9))
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.gcf().subplots_adjust(left=0.25)
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(labelsize=28)
    ax.yaxis.set_tick_params(labelsize=28)
    ax.xaxis.set_tick_params(length=10, width=5)
    ax.yaxis.set_tick_params(length=10, width=5)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(4)
    for tick in ax.xaxis.get_ticklabels():
        tick.set_fontname('Arial')
        tick.set_weight('bold')
    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontname('Arial')
        tick.set_weight('bold') 
        
    p0, = plt.plot(np.array(fitdGs), np.array(training_dG), 'bo', label = 'Final RMSE = %0.3f kcal/mol '% RMSD_fit + '(n=%0.0f)' %num_fit)

    plt.plot([-15,15], [-15,15], 'k--')
    plt.xlim([-1, 6])
    plt.ylim([-1, 6])
    plt.legend(handles=[p0], loc = 'upper left', fontsize = 16)
    plt.xlabel('$\Delta \Delta G_{Predicted}$', fontweight='bold', fontsize = 30, fontname = 'Arial')
    plt.ylabel('$\Delta \Delta G_{Observed}$', fontweight='bold', fontsize = 30, fontname = 'Arial')

    fig.savefig(save_name, dpi=400)



def fit_mutational_params(training_sequences, training_dG, testing_sequences, testing_dG, initial_mutation_penalties, mutation_range_high, mutation_range_low, initial_flip_params, save_prefix, Temperature, param_sensitivity = True):
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

    # Compute the fit ddG values for the training and testing sets
    fitdGs_training = func(results.params, np.array(list(training_sequences)))
    fitdGs_testing = func(results.params, np.array(list(testing_sequences)))

    # Plot ddG comparisions for training and test datasets
    ddG_comparison_plot(fitdGs_training, training_dG, save_prefix + 'ddG_comparison_training.pdf')
    ddG_comparison_plot(fitdGs_testing, testing_dG, save_prefix + 'ddG_comparison_testing.pdf')

    # Make parameter sensitivity plots
    if param_sensitivity:
        param_sensitivity_plots(results, func, params, param_names, Temperature, training_sequences, testing_sequences, save_prefix, training_dG, testing_dG)

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

    # define the initial flip penalties
    initial_flip_params = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])*2

    # define the save prefix
    savePrefix = './global_fit_ensemble_model_'

    # perform the fit
    fit_mutational_params(training_sequences, training_dG, testing_sequences, testing_dG, initial_params, mutation_range_high, mutation_range_low, initial_flip_params, savePrefix, Temperature, param_sensitivity = True)
    
    # close any plots that remain open
    matplotlib.pyplot.close("all")

main()

