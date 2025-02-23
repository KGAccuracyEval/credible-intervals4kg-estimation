#!/bin/bash

# path to evaluation procedure
python_file="./runKGEval.py"

# error rates
errorPs=(0.1 0.5 0.9)

for errorP in "${errorPs[@]}"; do
  # run file w/ wald configuration
  echo "run annotation process under Wald CI"
  python "$python_file" --dataset SYN --generator REM --errorP "$errorP" --method SRS --minSample 30 --ciMethod wald --confLevel 0.05
  python "$python_file" --dataset SYN --generator REM --errorP "$errorP" --method TWCS --minSample 30 --stageTwoSize 5 --ciMethod wald --confLevel 0.05
  python "$python_file" --dataset SYN --generator REM --errorP "$errorP" --method STWCS --minSample 30 --stageTwoSize 5 --numStrata 2 --stratFeature degree --ciMethod wald --confLevel 0.05

  # run file w/ wilson configuration
  echo "run annotation process under Wilson CI"
  python "$python_file" --dataset SYN --generator REM --errorP "$errorP" --method SRS --minSample 30 --ciMethod wilson --confLevel 0.05
  python "$python_file" --dataset SYN --generator REM --errorP "$errorP" --method TWCS --minSample 30 --stageTwoSize 5 --ciMethod wilson --confLevel 0.05
  python "$python_file" --dataset SYN --generator REM --errorP "$errorP" --method STWCS --minSample 30 --stageTwoSize 5 --numStrata 2 --stratFeature degree --ciMethod wilson --confLevel 0.05

  # bayesian credible intervals
  bayesCIs=(bayesET bayesHPD)
  priors=(0.3333333333333333 0.5 1 -1)  # 1/3 == Kerman prior, 1/2 == Jeffreys prior, 1 == uniform prior, -1 == adaptive strategy

  for bayesCI in "${bayesCIs[@]}"; do
    for prior in "${priors[@]}"; do
      echo "run annotation process under $bayesCI CI w/ prior $prior"
      python "$python_file" --dataset SYN --generator REM --errorP "$errorP" --method SRS --minSample 30 --ciMethod "$bayesCI" --alphaPrior "$prior" --betaPrior "$prior" --confLevel 0.05
      python "$python_file" --dataset SYN --generator REM --errorP "$errorP" --method TWCS --minSample 30 --stageTwoSize 5 --ciMethod "$bayesCI" --alphaPrior "$prior" --betaPrior "$prior" --confLevel 0.05
      python "$python_file" --dataset SYN --generator REM --errorP "$errorP" --method STWCS --minSample 30 --stageTwoSize 5 --numStrata 2 --stratFeature degree --ciMethod "$bayesCI" --alphaPrior "$prior" --betaPrior "$prior" --confLevel 0.05
    done
  done
done
