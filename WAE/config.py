import argparse, textwrap

formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=50)


def cla():
    parser = argparse.ArgumentParser(
        description="list of arguments", formatter_class=formatter
    )

    parser.add_argument(
        "--dataname",
        type=str,
        required=True,
        help=textwrap.dedent("""Name of the file containing the dataset"""),
    )
    parser.add_argument(
        "--slide_num",
        type=str,
        required=True,
        help=textwrap.dedent("""Slide identification number"""),
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=4000,
        help=textwrap.dedent(
            """Number of training samples to use. Cannot be more than that available."""
        ),
    )
    parser.add_argument(
        "--reg_param",
        type=float,
        default=1e-7,
        help=textwrap.dedent("""Regularization parameter"""),
    )
    parser.add_argument(
        "--n_epoch",
        type=int,
        default=1000,
        help=textwrap.dedent("""Maximum number of epochs"""),
    )
    parser.add_argument(
        "--z_dim",
        type=int,
        default=1,
        help=textwrap.dedent(
            """Dimension of the latent variable, when using type1 C-GAN"""
        ),
    )
    parser.add_argument(
        "--lambda_param",
        type=float,
        default=1.0,
        help=textwrap.dedent(
            """Beta weight on the KL regularization of the loss function"""
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help=textwrap.dedent("""Batch size while training"""),
    )
    parser.add_argument(
        "--learn_rate",
        type=float,
        default=1e-3,
        help=textwrap.dedent("""Initial Learning rate"""),
    )
    parser.add_argument(
        "--lr_sched",
        type=bool,
        default=True,
        help=textwrap.dedent("""Learning rate schedule"""),
    )
    parser.add_argument(
        "--act_function",
        type=str,
        default="ReLU",
        help=textwrap.dedent("""Activation function name"""),
    )
    parser.add_argument(
        "--denseblock_n",
        type=int,
        default=3,
        help=textwrap.dedent("""Number of blocks in denseblock"""),
    )

    parser.add_argument(
        "--seed_no",
        type=int,
        default=1008,
        help=textwrap.dedent("""Fix the random seed"""),
    )
    parser.add_argument(
        "--savefig_freq",
        type=int,
        default=100,
        help=textwrap.dedent(
            """Number of epochs after which a snapshot and plots are saved"""
        ),
    )
    parser.add_argument(
        "--save_suffix",
        type=str,
        default="",
        help=textwrap.dedent(
            """Suffix to directory where trained network/results are saved"""
        ),
    )
    parser.add_argument(
        "--sample_plots",
        type=int,
        default=5,
        help=textwrap.dedent("""Number of validation samples used to generate plots"""),
    )

    parser.add_argument(
        "--data_suffix",
        type=str,
        default="",
        help=textwrap.dedent(
            """Suffix to test results directory where results on test data are saved"""
        ),
    )
    parser.add_argument(
        "--main_dir",
        type=str,
        default="exps",
        help=textwrap.dedent(
            """Parent directory saving the various versions of trained networks"""
        ),
    )
    parser.add_argument(
        "--checkpoint_id",
        type=int,
        default=-1,
        help=textwrap.dedent("""The checkpoint index to load when testing"""),
    )

    return parser.parse_args()
