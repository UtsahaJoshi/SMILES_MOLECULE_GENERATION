import os
import pickle
import argparse

import fcd
from utils import canonicalize_smiles, getstats, loadmodel

def get_metric(trainset, submission, teststats, name):
    # Don't use GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Load training set for novelty
    with open(trainset, 'r') as f:
        smiles_train = {s.strip() for s in f if s.strip()}

    # Load submitted smiles. Only read 10000 smiles
    with open(submission, 'r') as f:
        smiles_gen = [s.strip() for s in f if s.strip()][:10000]

    smiles_can = canonicalize_smiles(smiles_gen)
    smiles_valid = [s for s in smiles_can if s is not None]
    smiles_unique = set(smiles_valid)
    smiles_novel = smiles_unique - smiles_train

    validity = len(smiles_valid) / len(smiles_gen)
    uniqueness = len(smiles_unique) / len(smiles_gen)
    novelty = len(smiles_novel) / len(smiles_gen)

    if name == 'validity':
        return validity
    elif name == 'uniqueness':
        return uniqueness
    elif name == 'novelty':
        return novelty
    elif name == 'fcd':
        # Load precomputed test mean and covariance
        with open(teststats, 'rb') as f:
            mean_test, cov_test = pickle.load(f)

        model = loadmodel()
        mean_gen, cov_gen = getstats(smiles_valid, model)

        fcd_value = fcd.calculate_frechet_distance(
            mu1=mean_gen, mu2=mean_test, sigma1=cov_gen, sigma2=cov_test)
        return fcd_value
    else:
        raise ValueError('Invalid metric specified')

def main():
    parser = argparse.ArgumentParser(description="Calculate metrics for SMILES.")
    parser.add_argument("--trainset", type=str, required=True, help="Path to the training set file")
    parser.add_argument("--submission", type=str, required=True, help="Path to the submission file")
    parser.add_argument("--teststats", type=str, required=True, help="Path to the test statistics file")
    parser.add_argument("--metric", type=str, choices=['validity', 'uniqueness', 'novelty', 'fcd'], required=True, help="The metric to calculate")

    args = parser.parse_args()

    result = get_metric(args.trainset, args.submission, args.teststats, args.metric)
    print(f"The {args.metric} is: {result:.4f}")

if __name__ == "__main__":
    main()
