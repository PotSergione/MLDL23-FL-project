import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--dataset', type=str, choices=['idda', 'femnist'], required=True, help='Dataset name')
    parser.add_argument('--niid', action='store_true', default=False,
                        help='Run the experiment with the non-IID partition (IID by default). Only on FEMNIST dataset.')
    parser.add_argument('--model', type=str, choices=['deeplabv3_mobilenetv2', 'resnet18', 'cnn', 'margnet', 'FedGen', 'FedRCN', 'FedADG', 'FedSR'], help='Model name')
    parser.add_argument('--num_rounds', type=int, help='Number of rounds')
    parser.add_argument('--num_epochs', type=int, help='Number of local epochs')
    parser.add_argument('--clients_per_round', type=int, help='Number of clients trained per round')
    parser.add_argument('--hnm', action='store_true', default=False, help='Use hard negative mining reduction or not')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--bs', type=int, default=4, help='Batch size')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay')
    parser.add_argument('--m', type=float, default=0.9, help='Momentum')
    parser.add_argument('--print_train_interval', type=int, default=100, help='Client print train interval')
    parser.add_argument('--print_test_interval', type=int, default=100, help='Client print test interval')
    parser.add_argument('--eval_interval', type=int, default=100, help='Eval interval')
    parser.add_argument('--test_interval', type=int, default=100, help='Test interval')

    # Added by our group to select the kind of smart client selection to perform
    parser.add_argument('--client_selection', type=str, choices=['uniform', 'smart10', 'smart30', 'poc', 'entropy'], default='uniform', 
                        help='Runs the experiment with different kind of client samplings')
    parser.add_argument('--d', type=int, choices=[2,5,8,10,20], default=20, 
                        help='Size of the candidate set for power of choice/entropy client selection')

    # Added by our group to select wether or not to launch a wandb session
    parser.add_argument('--wandb', type=str, choices=['on', 'off'], default='off', 
    help='Choose whether or not to launch a wandb session')

    # Added by our group to select wether or not to augment the data (with rotations)
    parser.add_argument('--augment', type=str, default='off', choices=['on', 'off'], 
    help='chooses whether to randomly rotate the data or not')

    # Added by our group to specify the angle to leave out
    parser.add_argument('--angle_out', type=int, default=75, choices=[0, 15, 30, 45, 60, 75], 
    help='Chooses the angle of rotation for the test clients')

    return parser
