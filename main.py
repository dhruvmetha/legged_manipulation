import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/apply_force.yaml')
    args = parser.parse_args()