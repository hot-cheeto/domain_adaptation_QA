import time

import argparse

def dump_parameters():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Experiment Parameters
    experiment_group = parser.add_argument_group("Experiment Configuration")
    experiment_group.add_argument("--experiment_id", default="{}".format(int(time.time())), 
                        help="experiment id is the name of that experiment run")
    experiment_group.add_argument("--load_checkpoint", default=False, action="store_true", 
                        help="If --do_learn is True, will load latest checkpoint, otherwise will load best.")
    experiment_group.add_argument("--note", default="", type=str,
                        help="Notes to write on the overall experiment spreadsheet about the expeirment run ")
    experiment_group.add_argument("--create_prediction_file", default=False, action="store_true", 
                        help = 'creates excel file with test set predictions')
    experiment_group.add_argument("--version", default=0, type=int,
                        help = 'version of checkpoint or experiment')
    experiment_group.add_argument("--epoch_key", default=-1, type=int,
                        help = 'which epoch to load')

    # Training Parameters
    training_group = parser.add_argument_group("Training")
    training_group.add_argument("--batch_size", default=64, type=int, 
                        help="Batch Size")
    training_group.add_argument("--num_workers", default=0, type=int, 
                        help="Number of workers to use when pre-fetching data.")
    training_group.add_argument("--num_epochs", default=20, type=int, 
                        help="Number of training epochs.")
    training_group.add_argument("--learning_rate", default=1e-3, type=float, 
                        help="Learning Rate")
    training_group.add_argument("--pin_memory", default=False, action="store_true", 
                        help="Whether or not memory should be pinned in PyTorch DataLoaders.")
    training_group.add_argument("--do_learn", default=False, action="store_true", 
                        help="Train the Network.")
    training_group.add_argument("--evaluate", default=False, action="store_true", 
                        help="Evaluate the Network on the test dataset.")
    training_group.add_argument("--evaluate_every", default=1, type=int,
                        help="Evaluate model on validation dataset every N epochs.")
    training_group.add_argument("--seed", default=43, type=int,
                        help="Seed to use for experiments. Good for reproducing results.")
    training_group.add_argument("--oversample", default=False, action = 'store_true',
                        help="oversample tgt dataset with joint training with teh src dataset")
    training_group.add_argument("--tgt_oversample_prob", default=0.7, type=float,
                        help = 'the prob when to sample tgt example during training')
    training_group.add_argument("--use_test_set", default=False, action = 'store_true',
                        help = 'for the bioasq dataset we have two test set one to obtain metrics and the other one to see what happen in real world example')
    training_group.add_argument("--oracle", default=False, action = 'store_true',
                        help = 'whihc is a bert cheater it trained on the test-dev and evaluate on the test-dev')

    # General Model Parameters
    model_general = parser.add_argument_group("General Model Parameters")
    model_general.add_argument("--dann", default=False, action = 'store_true', 
                        help="Use DANN")
    model_general.add_argument("--max_seq_length", default=384, type = int, 
                        help="input sequence max length of tokens")


    return parser.parse_args()
