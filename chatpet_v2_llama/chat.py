import argparse


def set_args():
    """
    Set arguments.
    1. device: GPU Setting
    2. temperature: Temperature for generation, default = 0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, required=False,
                        help='GPU Setting')
    parser.add_argument('--temperature', default=0, type=float, required=False,
                        help='Temperature for generation')