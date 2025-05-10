import numpy as np
from sklearn.neural_network import MLPRegressor  # Multi-layer Perceptron for predicting shapes
import gc

# clear memory cache
def clear_cache():
    gc.collect()

# Get model
def get_model(model_cache=None):
    if model_cache is None:
        model_cache = MLPRegressor(hidden_layer_sizes=(100,), max_iter=700, random_state=42, early_stopping=True, n_iter_no_change=20)
    return model_cache

def create_dim_pred(loaded_data):
    """
    Trains a model to predict output shapes from input shapes using training examples
    and then predicts the output shape for the last test case without a provided output.

    Args:
        loaded_data (dict): A dictionary representing a single ARC task,
                            containing 'train' and 'test' keys.

    Returns:
        tuple: (last_test_input_numpy_array, predicted_output_shape_tuple)
               Returns (None, None) if no suitable test case is found.
    """
    train_inputs_for_shape_model, train_outputs_for_shape_model = [], []

    # Collect train data for the shape predictor
    for example in loaded_data['train']:
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])

        # Use the first two dimensions (height, width)
        train_inputs_for_shape_model.append(input_grid.shape[:2])
        train_outputs_for_shape_model.append(output_grid.shape[:2])

    for test_case in loaded_data.get('test', []): # Use .get for safety
        if 'output' in test_case: # only if output is present
            input_grid = np.array(test_case['input'])
            output_grid = np.array(test_case['output'])
            train_inputs_for_shape_model.append(input_grid.shape[:2])
            train_outputs_for_shape_model.append(output_grid.shape[:2])

    if not train_inputs_for_shape_model:
        print("Warning: No training data available for shape prediction model.")
        # Fallback: try to predict output shape as same as input for the last test case
        # This is a heuristic and might not be accurate.
        last_unsolved_test_input_grid = None
        for test_case in reversed(loaded_data.get('test', [])):
            if 'output' not in test_case:
                last_unsolved_test_input_grid = np.array(test_case['input'])
                return last_unsolved_test_input_grid, last_unsolved_test_input_grid.shape[:2]
        return None, None # No unsolved test case found

    # Train a model to predict shapes
    shape_model = get_model()
    shape_model.fit(np.array(train_inputs_for_shape_model), np.array(train_outputs_for_shape_model)) # MLPRegressor expects 2D X, and y can be 2D for multi-output regression

    # Process test cases to find the one needing shape prediction
    last_unsolved_test_input_grid = None
    predicted_output_shape_tuple = None

    # Iterate in reverse to find the last test case that doesn't have an 'output'
    for test_case in reversed(loaded_data.get('test', [])):
        if 'output' not in test_case: # This is the one we need to predict for
            last_unsolved_test_input_grid = np.array(test_case['input'])
            # Predict using the first two dimensions of the input grid
            predicted_shape_array = shape_model.predict([last_unsolved_test_input_grid.shape[:2]])
            # predicted_shape_array will be like [[rows, cols]]
            # Round to nearest int and ensure positive dimensions
            predicted_output_shape_tuple = tuple(np.maximum(1, np.round(predicted_shape_array[0])).astype(int))
            break # Found the target test case

    if last_unsolved_test_input_grid is not None:
        # print("\n--- Predicted shape for last test input without output ---")
        # print(f"Input shape: {last_unsolved_test_input_grid.shape[:2]}")
        # print(f"Predicted output shape: {predicted_output_shape_tuple}")
        pass
    else:
        print("Warning: No unsolved test input found to predict shape for.")
        # Fallback if all test cases have outputs or no test cases.
        # This scenario should ideally not happen if create_arrays expects a prediction.
        # If there's a test input, predict its shape as same as input.
        if loaded_data.get('test'):
            last_input_arr = np.array(loaded_data['test'][-1]['input'])
            return last_input_arr, last_input_arr.shape[:2] # Fallback to input shape
        return None, None

    clear_cache()
    return last_unsolved_test_input_grid, predicted_output_shape_tuple