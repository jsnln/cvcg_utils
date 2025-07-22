import argparse
import shutil
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    path = parser.parse_args().path

    # working_directory = os.getcwd()
    file_directory = os.path.dirname(os.path.realpath(__file__))
    
    shutil.copytree(os.path.join(file_directory, 'basic_trainer'), path, ignore=shutil.ignore_patterns('__pycache__', 'results', 'wandb'), dirs_exist_ok=False)
