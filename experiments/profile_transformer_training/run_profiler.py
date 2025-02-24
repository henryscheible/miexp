import cProfile

from miexp.train.fourier import (  # noqa: F401
    FourierTrainingConfiguration,
    train_transformer_fourier,
)

if __name__ == "__main__":
    args = FourierTrainingConfiguration()

    cProfile.run("train_transformer_fourier(args)", "profile_results.txt")
