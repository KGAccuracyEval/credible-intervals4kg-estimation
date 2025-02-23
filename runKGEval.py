import pandas as pd
import numpy as np
import argparse
import random
import json
import os

from scipy import stats
from framework import samplingStrategies, labelGenerators


parser = argparse.ArgumentParser()

################################
###### Dataset parameters ######
################################

parser.add_argument('--dataset', default='YAGO', choices=['YAGO', 'NELL', 'DISGENET', 'FACTBENCH-MQ', 'DBPEDIA', 'SYN'], help='Target dataset.')
parser.add_argument('--generator', default='', choices=['REM', ''], help='Synthetic label generation model. Default to none.')
parser.add_argument('--errorP', default=0.1, type=float, help='Fixed error rate for synthetic label generation. Required by Random Model.')

###############################
###### Method parameters ######
###############################

parser.add_argument('--method', default='TWCS', choices=['SRS', 'TWCS', 'STWCS'], help='Method of choice.')
parser.add_argument('--minSample', default=30, type=int, help='Min sample size required to perform eval.')
parser.add_argument('--stageTwoSize', default=3, type=int, help='Second-stage sample size. Required by two-stage sampling methods.')
parser.add_argument('--ciMethod', default='wald', choices=['wald', 'wilson', 'bayesET', 'bayesHPD'], help='Methods to construct Confidence/Credible Intervals (CIs/CrIs).')
parser.add_argument('--alphaPrior', default=1, type=float, help='Parameter alpha used to setup Beta prior distribution -- only applies to bayesian credible intervals.')
parser.add_argument('--betaPrior', default=1, type=float, help='Parameter beta used to setup Beta prior distribution -- only applies to bayesian credible intervals.')

#######################################
###### Stratification parameters ######
#######################################

parser.add_argument('--numStrata', default=2, type=int, help='Number of strata considered by stratification based sampling methods.')
parser.add_argument('--stratFeature', default='degree', choices=['degree'], help='Stratification feature of choice.')

###################################
###### Estimation parameters ######
###################################

parser.add_argument('--confLevel', default=0.05, type=float, help='Estimator confidence level (1-confLevel).')
parser.add_argument('--thrMoE', default=0.05, type=float, help='Threshold for Margin of Error (MoE).')

###################################
###### Annotation parameters ######
###################################

parser.add_argument('--c1', default=45, type=int, help='Average cost for Entity Identification (EI).')
parser.add_argument('--c2', default=25, type=int, help='Average cost for Fact Verification (FV).')

##################################
###### Computing parameters ######
##################################

parser.add_argument('--iterations', default=1000, type=int, help='Number of iterations for computing estimates.')
parser.add_argument('--seed', default=42, type=int, help='Random seed.')

args = parser.parse_args()


def labelGenerator(method):
    """

    :param method: synthetic label generation model
    :return: instance of specified synthetic label generation model
    """

    return {
        'REM': lambda: labelGenerators.RandomErrorModel(),
    }[method]()


def samplingMethod(method, confLevel, alphaPrior, betaPrior):
    """
    Instantiate the specific sampling method

    :param method: sampling method
    :param confLevel: estimator confidence level (1-confLevel)
    :param alphaPrior: alpha parameter used for Beta prior distribution (only applies to bayesian credible intervals)
    :param betaPrior: beta parameter used for Beta prior distribution (only applies to bayesian credible intervals)
    :return: instance of specified sampling method
    """

    return {
        'SRS': lambda: samplingStrategies.SRSSampler(confLevel, alphaPrior, betaPrior),
        'TWCS': lambda: samplingStrategies.TWCSSampler(confLevel, alphaPrior, betaPrior),
        'STWCS': lambda: samplingStrategies.STWCSSampler(confLevel, alphaPrior, betaPrior)
    }[method]()


def main():
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    print('Load {} dataset.'.format(args.dataset))
    # get target dataset
    if args.dataset == 'FACTBENCH-MQ':
        with open('./dataset/FACTBENCH/data/kg.json', 'r') as f:
            id2triple = json.load(f)
    else:
        with open('./dataset/' + args.dataset + '/data/kg.json', 'r') as f:
            id2triple = json.load(f)

    # set KG as [(id, triple), ...]
    if args.dataset == 'FACTBENCH-MQ':
        kg = [(k, v[0]) for k, v in id2triple.items()]
        kg = [p for p in kg if
              'correct_' in p[0] or
              'wrong_mix_domain' in p[0] or
              'wrong_mix_range' in p[0] or
              'wrong_mix_domainrange' in p[0] or
              'wrong_mix_property' in p[0] or
              'wrong_mix_random' in p[0]]
    else:
        kg = list(id2triple.items())

    if args.generator:  # generate synthetic labels
        # set generator to label the KG
        print('Set {} generator to label {} KG'.format(args.generator, args.dataset))
        generator = labelGenerator(args.generator)
        # set params to generate labels
        gParams = {'kg': kg}
        if args.generator == 'REM':  # REM generator requires fixed error rate
            print('{} w/ error rate = {}'.format(args.generator, args.errorP))
            gParams['errorP'] = args.errorP
        # annotate KG w/ generator
        gt = generator.annotateKG(**gParams)
    else:  # target dataset has ground truth
        if args.dataset == 'FACTBENCH-MQ':
            with open('./dataset/FACTBENCH/data/gt.json', 'r') as f:  # get ground truth
                gt = json.load(f)
        else:
            with open('./dataset/' + args.dataset + '/data/gt.json', 'r') as f:  # get ground truth
                gt = json.load(f)

        if args.dataset == 'FACTBENCH-MQ':
            gt = {k: v for k, v in gt.items() if k in dict(kg)}

    # compute KG (real) accuracy
    acc = sum(gt.values())/len(gt)
    print('KG (real) accuracy: {}'.format(acc))

    # set efficient KG accuracy estimator w/ confidence level 1-args.confLevel
    print('Set {} estimator with confidence level {}%.'.format(args.method, 1 - args.confLevel))
    estimator = samplingMethod(args.method, args.confLevel, args.alphaPrior, args.betaPrior)

    # set params to perform evaluation
    eParams = {'kg': kg, 'groundTruth': gt, 'minSample': args.minSample, 'thrMoE': args.thrMoE, 'ciMethod': args.ciMethod, 'iters': args.iterations}
    if (args.method == 'TWCS') or (args.method == 'STWCS'):  # two-stage sampling methods require second-stage sample size parameter
        eParams['stageTwoSize'] = args.stageTwoSize
    if args.method == 'STWCS':  # stratified sampling methods require stratification feature and warmup number of triples
        eParams['numStrata'] = args.numStrata
        eParams['stratFeature'] = args.stratFeature

    eParams['c1'] = args.c1
    eParams['c2'] = args.c2

    # perform the evaluation procedure for args.iterations times and compute estimates
    print('Perform KG accuracy evaluation for {} times and stop at each iteration when MoE < {}'.format(args.iterations, args.thrMoE))
    estimates = estimator.run(**eParams)
    # convert estimates to numpy and print results
    estimates = pd.DataFrame(estimates, columns=['annotTriples', 'estimatedAcc', 'annotCost', 'lowerBound', 'upperBound'])
    print('estimated accuracy: mean={} stdev={}'.format(estimates['estimatedAcc'].mean(), estimates['estimatedAcc'].std()))
    print('annotated triples: mean={} stdev={}'.format(estimates['annotTriples'].mean(), estimates['annotTriples'].std()))
    print('annotation cost (hours): mean={} stdev={}'.format(estimates['annotCost'].mean(), estimates['annotCost'].std()))
    
    # create dir (if not exists) where storing estimates
    if args.dataset == 'FACTBENCH-MQ':
        dname = './results/FACTBENCH/'+args.dataset.split('-')[-1]+'/'
    else:
        dname = './results/'+args.dataset+'/'
    if args.generator:
        dname += args.generator+'/'+args.ciMethod+'/'
        if args.ciMethod in ['bayesET', 'bayesHPD']:
            dname += 'a='+str(round(args.alphaPrior, 2))+'_b='+str(round(args.betaPrior, 2))+'/'

        if args.generator == 'REM':
            dname += str(args.errorP)+'/'
    else:
        dname += args.ciMethod+'/'
        if args.ciMethod in ['bayesET', 'bayesHPD']:
            dname += 'a='+str(round(args.alphaPrior, 2))+'_b='+str(round(args.betaPrior, 2))+'/'
    os.makedirs(dname, exist_ok=True)
    # set file name
    fname = args.method + '_batch=' + str(args.minSample)
    if args.method in ['TWCS', 'STWCS']:
        fname += '_stage2=' + str(args.stageTwoSize)
    if args.method == 'STWCS':
        fname += '_feature=' + args.stratFeature + '_strata=' + str(args.numStrata)
    # add file type
    fname += '.tsv'
    # store estimates
    estimates.to_csv(dname+fname, sep='\t', index=False)
    print('Estimates stored in {}{}'.format(dname, fname))


if __name__ == "__main__":
    main()
