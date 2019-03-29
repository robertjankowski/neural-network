import sys
import os
import subprocess
from subprocess import Popen, PIPE


def main():

    cmd = ['./neuralnet', '100', '2', '0.1']

    outFile = open("file.txt", 'a+')

    result = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    out = result.stdout.read()

    outFile.write(str(out))
    outFile.close()

    # epochs = sys.argv[1]
    # minibatch_size = sys.argv[2]
    # eta = sys.argv[3]
    # file = "./neuralnet " + epochs + " " + minibatch_size + " " + eta
    # os.system(file)


if __name__ == "__main__":
    main()
