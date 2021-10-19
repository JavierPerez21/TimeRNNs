import torch

data = ["hil-clean-09.csv", "hil-clean-05.csv"]
# Initialize some variables
datapath = data
global TIME_EMBED_SIZE
TIME_EMBED_SIZE = 10
global BATCHSIZE
BATCHSIZE = 2000
global INPUT_LENGTH
INPUT_LENGTH = 25
global OUTPUT_LENGTH
OUTPUT_LENGTH = 50
global LONGER_LENGTH
LONGER_LENGTH = INPUT_LENGTH
if OUTPUT_LENGTH > INPUT_LENGTH:
  LONGER_LENGTH = OUTPUT_LENGTH
num_epochs = 80
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")