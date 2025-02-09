from miexp.script_util import parse_args_from_conf
from miexp.train.fourier import FourierTrainingConfiguration, train_transformer_fourier

if __name__ == "__main__":
    args = parse_args_from_conf(FourierTrainingConfiguration)
    train_transformer_fourier(args)
