import argparse

from src.trainer import trainer
from src.tester import tester

def main():
    parser = argparse.ArgumentParser(description='Train or test SAC algorithm')

    parser.add_argument('--train',
                        help='trains SAC algorithm',
                        action='store_true')
    
    parser.add_argument('--test',
                        help='test SAC algorithm')

    args = parser.parse_args()

    if args.train and args.test:
        parser.error("Specify just one argument.")

    if args.train:
        trainer()
    elif args.test:
        tester()

    else:
        parser.error("Specify at least one argument: --train or --test.")


if __name__ == "__main__":
    main()