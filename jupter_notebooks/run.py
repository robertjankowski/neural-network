import sys
import os


def main():
    epochs = sys.argv[1]
    minibatch_size = sys.argv[2]
    eta = sys.argv[3]
    file = "../bin/neuralnet " + epochs + " " + minibatch_size + " " + eta
    os.system(file)


if __name__ == "__main__":
    main()
