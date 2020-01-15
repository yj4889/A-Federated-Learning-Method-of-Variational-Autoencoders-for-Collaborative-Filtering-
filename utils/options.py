import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--data_dir', type=str, default='ml-20m', help='Movielens-20m dataset location')
    parser.add_argument('--preprocess', type=int, default=0, help='1 for preprocessing, 0 for no preprocessing')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size')
    parser.add_argument('--batch_size_vad', type=int, default=2000, help='validation batch size')
    parser.add_argument('--total_anneal_steps', type=int, default=200000, help='number of total anneal steps')
    parser.add_argument('--anneal_cap', type=int, default=0.2, help='largest annealing parameter')
    parser.add_argument('--n_epochs', type=int, default=50, help='training epochs')

    parser.add_argument('--data_dir2', type=str, default='ml-20m/pro_sg/unique_sid.txt', help='Movielens-20m dataset location')
    args = parser.parse_args()
    return args