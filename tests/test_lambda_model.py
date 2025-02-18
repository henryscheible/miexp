import torch

from miexp.util.lambda_model import LambdaModel


def test_lambda_model_forward():
    # Define a simple function to be used with LambdaModel
    def simple_function(x):
        return x * 2

    # Create an instance of LambdaModel with the simple function
    model = LambdaModel(simple_function)

    # Test the forward method with a sample input
    input_value = 3
    expected_output = 6
    assert model(input_value) == expected_output


def test_lambda_model_forward_with_kwargs():
    # Define a function that uses keyword arguments
    def function_with_kwargs(x, factor=1):
        return x * factor

    # Create an instance of LambdaModel with the function
    model = LambdaModel(function_with_kwargs)

    # Test the forward method with keyword arguments
    input_value = 3
    factor = 4
    expected_output = 12
    assert model(input_value, factor=factor) == expected_output


def test_lambda_model_forward_with_multiple_args():
    # Define a function that uses multiple arguments
    def function_with_multiple_args(x, y):
        return x + y

    # Create an instance of LambdaModel with the function
    model = LambdaModel(function_with_multiple_args)

    # Test the forward method with multiple arguments
    input_value1 = 3
    input_value2 = 4
    expected_output = 7
    assert model(input_value1, input_value2) == expected_output


def test_put_lambda_model_on_device():
    def simple_function(x):
        return x * 2

    # Create an instance of LambdaModel with the simple function
    model = LambdaModel(simple_function)
    model.to(torch.device("cpu"))
