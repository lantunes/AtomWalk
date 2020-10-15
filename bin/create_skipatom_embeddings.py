import sys
sys.path.append('../atomwalk')

from skipatom import TrainingData, Trainer
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str,
                        help='The path to the training data file.')
    parser.add_argument('--out', '-o', type=str,
                        help='The path to the embeddings file to be created.')
    parser.add_argument('--dim', type=int,
                        help='The number of embedding dimensions.')
    parser.add_argument('--step', type=float, default=0.01,
                        help='The step size.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='The number of epochs.')
    parser.add_argument('--batch', type=int, default=256,
                        help='The batch size.')
    args = parser.parse_args()

    td = TrainingData.load(args.data)

    target_atoms, context_atoms = td.to_one_hot()

    tr = Trainer(dim_in=args.dim, dim_out=len(context_atoms[0]))

    embeddings = tr.train(target_atoms, context_atoms, step_size=args.step, num_epochs=args.epochs, batch_size=args.batch)

    Trainer.save_embeddings(embeddings, args.out)
