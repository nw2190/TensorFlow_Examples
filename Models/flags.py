from argparse import ArgumentParser

# Specify default arguments common to all models
def add_default_args(parser):

    # Default training options
    parser.add_argument("--training_steps", default=10000, type=int, help="Total number of training steps")
    parser.add_argument("--batch_size", default=64, type=int, help="Training batch size")
    parser.add_argument("--learning_rate", default=0.0002, type=float, help="Initial learning rate")
    parser.add_argument("--lr_decay_step", default=1100, type=int, help="Learning rate decay step")
    parser.add_argument("--lr_decay_rate", default=0.75, type=float, help="Learning rate decay rate")
    parser.add_argument("--adam_beta1", default=0.5, type=float, help="Adam optimizer beta1 parameter")
    
    # Saving and display options
    parser.add_argument("--display_step", default=1, type=int, help="Step count for displaying progress")
    parser.add_argument("--summary_step", default=100, type=int, help="Step count for saving summaries")
    parser.add_argument("--log_dir", default="./Model/logs/", type=str, help="Directory for saving log files")
    parser.add_argument("--checkpoint_step", default=250, type=int, help="Step count for saving checkpoints")
    parser.add_argument("--checkpoint_dir", default="./Model/Checkpoints/", type=str, help="Directory for saving checkpoints")
    parser.add_argument("--plot_step", default=250, type=int, help="Step count for saving plots of generated images")
    parser.add_argument("--plot_dir", default="./Model/predictions/", type=str, help="Directory for saving plots of generated images")
    parser.add_argument("--plot_res", default=64, type=int, help="Resolution to use when saving generated images")
    
    return parser

# Define flags to specify model hyperparameters and training options for GAN Model
def getFlags_GAN():

    # Initialize argument parser and add default arguments
    parser = ArgumentParser(description='Argument Parser')
    parser = add_default_args(parser)
    
    # GAN hyperparameters
    parser.add_argument("--z_dim", default=62, type=int, help="Dimension of noise vector in latent space")
    parser.add_argument("--g_res", default=7, type=int, help="Resolution of initial reshaped features in generator")
    parser.add_argument("--g_chans", default=128, type=int, help="Channel count for initial reshaped features in generator")

    # Parse arguments from command line
    args = parser.parse_args()
    return args


# Define flags to specify model hyperparameters and training options for VAE Model
def getFlags_VAE():

    # Initialize argument parser and add default arguments
    parser = ArgumentParser(description='Argument Parser')
    parser = add_default_args(parser)
    
    # VAE hyperparameters
    parser.add_argument("--z_dim", default=62, type=int, help="Dimension of noise vector in latent space")
    parser.add_argument("--min_res", default=7, type=int, help="Resolution of initial reshaped features in encoder/decoder")
    parser.add_argument("--min_chans", default=128, type=int, help="Channel count for initial reshaped features in encoder/decoder")
    parser.add_argument("--early_stopping_start", default=2000, type=int, help="Starting step for early stopping checks")
    parser.add_argument("--early_stopping_step", default=1000, type=int, help="Steps between early stopping checks")
    parser.add_argument("--early_stopping_tol", default=50.0, type=float, help="Tolerance for early stopping")

    # Parse arguments from command line
    args = parser.parse_args()
    return args
