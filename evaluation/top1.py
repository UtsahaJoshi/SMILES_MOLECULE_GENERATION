import argparse
import pandas as pd
from rdkit import Chem

def process_chemicals(chemicals):
    """
    Preprocess a list of chemical SMILES strings using an external library,
    ensuring they are valid and canonicalized.
    """
    processed_chemicals = []
    for chem in chemicals:
        m = Chem.MolFromSmiles(chem)
        if m is not None:
            processed_chemicals.append(Chem.MolToSmiles(m))
    return processed_chemicals

def calculate_top_1_accuracy(predictions_file_path, true_answers_file_path):
    
    # Open the files
    df_preds = pd.read_csv(predictions_file_path, header=None, names=['prediction'])
    df_true = pd.read_csv(true_answers_file_path, header=None, names=['true_reactants'])

    # Assert the files have the same number of lines
    assert len(df_preds) == len(df_true), "Mismatch in number of lines between predictions and true answers."

    # Merge predictions and true labels
    df = pd.concat([df_preds, df_true], axis=1)
    # Drop rows without true labels
    df = df.dropna()
    
    correct_predictions = 0

    # Process each line of the files
    for pred, true in zip(df.prediction, df.true_reactants):
    
        pred = pred.strip()
        true = true.strip()

        # Preprocess true and predicted reagents
        true_reagents_set = set(process_chemicals(true.split('.')))
        pred_reagents_set = set(process_chemicals(pred.split('.')))

        # Compare the sets
        if true_reagents_set == pred_reagents_set:
            correct_predictions += 1

    # Calculate and return top-1 accuracy
    top_1_accuracy = correct_predictions / df.shape[0]
    return top_1_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()
    print(calculate_top_1_accuracy(args.submission, args.target))