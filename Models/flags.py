from argparse import ArgumentParser

# Define flags to specify model hyperparameters and training options
def getFlags_GAN():

    # Initialize argument parser
    parser = ArgumentParser(description='Argument Parser')

    # Specify arguments and default values for parser
    parser.add_argument("--training_steps", default=10000, type=int, help="Total number of training steps")
    parser.add_argument("--batch_size", default=64, type=int, help="Training batch size")
    parser.add_argument("--learning_rate", default=0.0002, type=float, help="Initial learning rate")
    parser.add_argument("--lr_decay_step", default=1100, type=int, help="Learning rate decay step")
    parser.add_argument("--lr_decay_rate", default=0.75, type=float, help="Learning rate decay rate")
    parser.add_argument("--adam_beta1", default=0.5, type=float, help="Adam optimizer beta1 parameter")
    
    # GAN hyperparameters
    parser.add_argument("--z_dim", default=62, type=int, help="Dimension of noise vector in latent space")
    parser.add_argument("--g_res", default=7, type=int, help="Resolution of initial reshaped features in generator")
    parser.add_argument("--g_chans", default=128, type=int, help="Channel count for initial reshaped features in generator")
    parser.add_argument("--l2_reg_scale", default=0.0000001, type=float, help="L2 weight regularization scale")
    parser.add_argument("--early_stopping_tol", default=0.0025, type=int, help="Tolerance for early stopping")

    # Saving and display options
    parser.add_argument("--display_step", default=1, type=int, help="Step count for displaying progress")
    parser.add_argument("--summary_step", default=100, type=int, help="Step count for saving summaries")
    parser.add_argument("--log_dir", default="./Model/logs/", type=str, help="Directory for saving log files")
    parser.add_argument("--checkpoint_step", default=250, type=int, help="Step count for saving checkpoints")
    parser.add_argument("--checkpoint_dir", default="./Model/Checkpoints/", type=str, help="Directory for saving checkpoints")
    parser.add_argument("--plot_step", default=250, type=int, help="Step count for saving plots of generated images")
    parser.add_argument("--plot_dir", default="./Model/predictions/", type=str, help="Directory for saving plots of generated images")
    parser.add_argument("--plot_res", default=64, type=int, help="Resolution to use when saving generated images")

    # Parse arguments from command line
    args = parser.parse_args()
    return args
