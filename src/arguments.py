import time

import argparse

def dump_parameters():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # NOTE: If you want to add your own model, add it to the "choices" list:
    # parser.add_argument("--model_type", type=str, default="attention_cn2a", 
                        # choices=["attention_cn2a", "iur", "iur_conditioning"],
                        # help="Specifies which model to use.")

    # Experiment Parameters
    experiment_group = parser.add_argument_group("Experiment Configuration")
    experiment_group.add_argument("--dataset_name", default="author_100", type=str,
                        help="Datset used for training and loading the models.")
    experiment_group.add_argument("--experiment_id", default="{}".format(int(time.time())), 
                        help="experiment id is the name of that experiment run")
    experiment_group.add_argument("--load_checkpoint", default=False, action="store_true", 
                        help="If --do_learn is True, will load latest checkpoint, otherwise will load best.")
    experiment_group.add_argument("--note", default="", type=str,
                        help="Notes to write on the overall experiment spreadsheet about the expeirment run ")

    # Training Parameters
    training_group = parser.add_argument_group("Training")
    training_group.add_argument("--batch_size", default=64, type=int, 
                        help="Batch Size")
    training_group.add_argument("--num_workers", default=0, type=int, 
                        help="Number of workers to use when pre-fetching data.")
    training_group.add_argument("--num_epochs", default=100, type=int, 
                        help="Number of training epochs.")
    training_group.add_argument("--learning_rate", default=1e-3, type=float, 
                        help="Learning Rate")
    training_group.add_argument("--retrieval", default=False, action="store_true", 
                        help="If True, will act in the open-world scenario.")
    training_group.add_argument("--pin_memory", default=False, action="store_true", 
                        help="Whether or not memory should be pinned in PyTorch DataLoaders.")
    training_group.add_argument("--do_learn", default=False, action="store_true", 
                        help="Train the Network.")
    training_group.add_argument("--extract_emb", default=False, action="store_true", 
                        help="Extract feature embeddings.")
    training_group.add_argument("--evaluate", default=False, action="store_true", 
                        help="Evaluate the Network on the test dataset.")
    training_group.add_argument("--loss", default="cross_entropy", 
                        choices=["cross_entropy", "arcface_loss", "triplet_loss"],
                        help="Loss used to train the network.")
    training_group.add_argument("--evaluate_every", default=1, type=int,
                        help="Evaluate model on validation dataset every N epochs.")
    training_group.add_argument("--seed", default=43, type=int,
                        help="Seed to use for experiments. Good for reproducing results.")

    # General Model Parameters
    model_general = parser.add_argument_group("General Model Parameters")
    model_general.add_argument("--author_embedding_dim", default=512, type=int, 
                        help="Author embedding dimension.")
    model_general.add_argument("--cont_parent_text", default=False, action='store_true', 
                        help="Conditioning on parent text.")
    model_general.add_argument("--cont_parent_author", default=False, action='store_true', 
                        help="Conditioning on parent author.")
    model_general.add_argument("--cont_parent_time", default=False, action='store_true', 
                        help="Conditioning on parent time.")


    return parser.parse_args()
