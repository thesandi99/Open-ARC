from untils import create_dim_pred
import numpy as np
import argparse
import json

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Dimension Predictor")
    parser.add_argument('--path', type=str, default=None, help="Path to test data")
    parser.add_argument('--value', type=str, default=None, help="Additional value for the test data")
    
    args = parser.parse_args()

    if args.value is not None:
        Warning("Value is not used in the current implementation. It will be ignored.")

    # Handle path input
    if args.path is None:
        print("Path to test data must be provided. Use --path <path>. Current Use Default Value To add Value Use --value <value>")
        test_data = """{"train":[{"input":[[0,3,0,0,0,2,0,0,0,0,0,0,0,0,0,0],[0,3,3,3,0,2,0,0,0,0,0,0,0,0,0,0],[3,3,0,0,0,2,0,0,0,0,0,0,0,0,0,0],[0,3,3,3,0,2,0,0,0,0,0,0,0,0,0,0]],"output":[[0,0,0,0,0,2,0,3,0,0,0,0,0,0,0,0],[0,0,0,0,0,2,0,3,3,3,0,0,0,0,0,0],[0,0,0,0,0,2,3,3,0,0,0,0,0,0,0,0],[0,0,0,0,0,2,0,3,3,3,0,0,0,0,0,0]]},{"input":[[0,0,0,0,0],[3,3,0,0,0],[3,0,0,0,0],[3,3,0,3,3],[0,3,3,3,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[2,2,2,2,2],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],"output":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[2,2,2,2,2],[3,3,0,0,0],[3,0,0,0,0],[3,3,0,3,3],[0,3,3,3,0],[0,0,0,0,0],[0,0,0,0,0]]},{"input":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[2,2,2,2,2],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[3,3,3,3,0],[3,0,0,3,0],[3,3,0,3,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],"output":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[3,3,3,3,0],[3,0,0,3,0],[3,3,0,3,0],[2,2,2,2,2],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]}],"test":[{"input":[[0,0,0,0,0,0,0,0,2,0,0,0,3,3,0,0,0,0],[0,0,0,0,0,0,0,0,2,0,0,3,0,3,0,0,0,0],[0,0,0,0,0,0,0,0,2,0,0,3,3,3,0,0,0,0],[0,0,0,0,0,0,0,0,2,0,0,3,0,0,0,0,0,0]]}]}"""
        test_data = json.loads(test_data)
    else:
        # Load the test data
        with open(args.path, 'r') as f:
            test_data = json.load(f)

    # Create dimension predictor
    dim_pred = create_dim_pred(test_data)
    
    # Print the shape of the dimension predictor
    print("Shape of the dimension predictor:", dim_pred)

if __name__ == "__main__":
    main()

# !python test/dim_prediction.py 