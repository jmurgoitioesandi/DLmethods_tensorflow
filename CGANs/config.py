import argparse, textwrap

formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=50)

def cla():
    parser = argparse.ArgumentParser(description='list of arguments',formatter_class=formatter)

    parser.add_argument('--problem'   , type=str, required=True, help=textwrap.dedent('''Joint distribution being learned'''))
    parser.add_argument('--max_epochs'   , type=int, required=True, help=textwrap.dedent('''Max. number of epochs'''))
    parser.add_argument('--repetition'   , type=str, required=True, help=textwrap.dedent('''Indicator for the folder name'''))
    parser.add_argument('--loss_type'   , type=str, required=True, help=textwrap.dedent('''Type of loss used, WGAN vs ctransform'''))
    return parser.parse_args()


