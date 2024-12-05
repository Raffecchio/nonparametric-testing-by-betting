"""
Experiment 1 from Section VI: "Numerical Experiments"
Comparison of power of the sequential tests 
"""
import os
import pickle 
import argparse
from functools import partial
from math import sqrt
from time import time 

import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from utils import RBFkernel, runBatchTwoSampleTest
from sources import GaussianSource, getGaussianSourceparams

from kernelMMD import computeMMD, kernelMMDprediction
from SeqTestsUtils import runSequentialTest, ONSstrategy, KellyBettingApprox
from SeqLC import runLCexperiment 
from SeqOther import runLilTest, runMRTest


def main(N_betting=600, d=10, num_trials=20,
            alpha=0.05, epsilon_mean=0.35, epsilon_var=1.5, num_pert_mean=1, 
            num_pert_var=0, progress_bar=False):
   
    # Maximum sample size 
    meanX, meanY, covX, covY = getGaussianSourceparams(d=d, epsilon_mean=epsilon_mean, 
                                    epsilon_var=epsilon_var,
                                    num_perturbations_mean=num_pert_mean, 
                                    num_perturbations_var=num_pert_var)
    # generate the source 
    Source = GaussianSource(meanX=meanX, meanY=meanY, 
                            covX=covX, covY=covY, truncated=False)

    # Do the betting based sequential kernel-MMD test 
    print(f'\n====Starting Betting Sequential test====\n')
    t0 = time()
    Prediction = kernelMMDprediction
    Betting = ONSstrategy
    pred_params=None 
    bet_params=None
    PowerBetting, StoppedBetting, StoppingTimesBetting = runSequentialTest(Source, Prediction, Betting, 
                                                            alpha=alpha, Nmax=N_betting,
                                                            pred_params=pred_params, bet_params=bet_params,
                                                            num_trials=num_trials, progress_bar=True)
    mean_tau_betting = StoppingTimesBetting.mean()
    NNBetting = np.arange(1, N_betting+1)
    deltaT = time() - t0 
    print(f'Sequential Betting test took {deltaT:.2f} seconds')
    ## Prepare the data for plotting 
    Data = {}
    Data['betting']=(PowerBetting, NNBetting, mean_tau_betting, StoppingTimesBetting) 

    return Data 


def plot_results(Data, title, xlabel, ylabel, savefig=False, figname=None):
    palette = sns.color_palette(palette='husl', n_colors=10)
    plt.figure() 
    i=0
    for method in Data:
        if method=='batch':
            power, NN = Data[method] 
            plt.plot(NN, power, label=method, color=palette[i])
        else:
            power, NN, mean_tau, _ = Data[method]
            plt.plot(NN, power, label=method, color=palette[i])
            plt.axvline(x=mean_tau, linestyle='--', color=palette[i])
        i+=1 
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=15)
    plt.legend(fontsize=12)
    if savefig:
        figname = 'temp.png' if figname is None else figname 
        plt.savefig(figname, dpi=450)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dim', '-d', default=1, type=int, help='data dimension')
    parser.add_argument('--num_obs', '-nobs', default=600, type=int, help='number of observations')
    parser.add_argument('--eps_mean', '-em', default=0.5, type=float)
    parser.add_argument('--eps_var', '-ev', default=0, type=float)

    parser.add_argument('--num_trials', '-nt', default=500, type=int)
    parser.add_argument('--save_fig', '-sf', action='store_true')
    parser.add_argument('--save_data', '-sd', action='store_true')
    parser.add_argument('--alpha', '-a', default=0.05, type=float)

    parser.add_argument('--name', '-n', default="Experiment_simplebet", type=str)

    args = parser.parse_args()

    d = args.data_dim #10
    epsilon_mean = args.eps_mean #0.5
    epsilon_var = args.eps_var #0.5
    N_betting = args.num_obs

    num_trials = args.num_trials #500
    alpha = args.alpha #0.05

    ### parameters of the Gaussian distribution 
    num_pert_mean = 1 
    num_pert_var = 1

    # Flags to save the data 
    savefig=args.save_fig
    savedata=args.save_data

    # run the experiment
    DataToPlot = main(N_betting=N_betting, d=d,
                    num_trials=num_trials,
                    alpha=alpha, epsilon_mean=epsilon_mean,
                    epsilon_var=epsilon_var, num_pert_mean=num_pert_mean, 
                    num_pert_var=num_pert_var,
                    progress_bar=True)
    title='Power vs Sample Size'
    xlabel='Sample-Size (n)'
    ylabel='Power'
    # get the path of the file to store data 
    parent_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    data_dir = parent_dir + '/data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    name = args.name
    figname = f'{data_dir}/' + name + '.png'
    plot_results(DataToPlot, title, xlabel, ylabel, savefig=savefig, figname=figname)

    filename = f'{data_dir}/Experiment1data.pkl'
    with open(filename, 'wb') as handle: 
        pickle.dump(DataToPlot, handle)

