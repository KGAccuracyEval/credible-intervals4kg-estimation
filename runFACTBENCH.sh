#!/bin/bash

# path to evaluation procedure
python_file="./runKGEval.py"

# run file w/ wald configuration
echo "run annotation process under Wald CI"
python "$python_file" --dataset FACTBENCH-MQ --method SRS --minSample 30 --ciMethod wald --confLevel 0.05
python "$python_file" --dataset FACTBENCH-MQ --method TWCS --minSample 30 --stageTwoSize 3 --ciMethod wald --confLevel 0.05
python "$python_file" --dataset FACTBENCH-MQ --method STWCS --minSample 30 --stageTwoSize 3 --numStrata 4 --stratFeature degree --ciMethod wald --confLevel 0.05

# run file w/ wilson configuration
echo "run annotation process under Wilson CI"
python "$python_file" --dataset FACTBENCH-MQ --method SRS --minSample 30 --ciMethod wilson --confLevel 0.05
python "$python_file" --dataset FACTBENCH-MQ --method TWCS --minSample 30 --stageTwoSize 3 --ciMethod wilson --confLevel 0.05
python "$python_file" --dataset FACTBENCH-MQ --method STWCS --minSample 30 --stageTwoSize 3 --numStrata 4 --stratFeature degree --ciMethod wilson --confLevel 0.05

# bayesian credible intervals
bayesCIs=(bayesET bayesHPD)
priors=(0.3333333333333333 0.5 1 -1)  # 1/3 == Kerman prior, 1/2 == Jeffreys prior, 1 == uniform prior, -1 == adaptive strategy

for bayesCI in "${bayesCIs[@]}"; do
  for prior in "${priors[@]}"; do
    echo "run annotation process under $bayesCI CI w/ prior $prior"
    python "$python_file" --dataset FACTBENCH-MQ --method SRS --minSample 30 --ciMethod "$bayesCI" --alphaPrior "$prior" --betaPrior "$prior" --confLevel 0.05
    python "$python_file" --dataset FACTBENCH-MQ --method TWCS --minSample 30 --stageTwoSize 3 --ciMethod "$bayesCI" --alphaPrior "$prior" --betaPrior "$prior" --confLevel 0.05
    python "$python_file" --dataset FACTBENCH-MQ --method STWCS --minSample 30 --stageTwoSize 3 --numStrata 4 --stratFeature degree --ciMethod "$bayesCI" --alphaPrior "$prior" --betaPrior "$prior" --confLevel 0.05
  done
done
