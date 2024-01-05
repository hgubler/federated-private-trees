# Federated Differentially Private Decision Tree

This repository contains the code for a semester project focused on designing a federated differentially private decision tree algorithm.

## Project Structure

The project is structured into two main directories: `algorithms` and `experiments`.

### Algorithms

The `algorithms` directory contains the implementation of the federated differentially private decision tree algorithm. The main file in this directory is `federated_private_decision_tree.py`.

This algorithm can be run with and without privacy budget saving. This is controlled by the `save_budget` parameter. When `save_budget` is set to `True`, the algorithm uses feature impurity bounds to save privacy budget.

### Experiments

The `experiments` directory contains the code used to create the plots for our simulation study, as detailed in the project report. This includes various scripts for running the algorithm under different conditions and visualizing the results.

## Usage

To run the federated differentially private decision tree algorithm, navigate to the `algorithms` directory and run the `federated_private_decision_tree.py` file. You can control whether or not the algorithm saves privacy budget by setting the `save_budget` parameter.

To reproduce the experiments from the report, navigate to the `experiments` directory and run the corresponding scripts.

## Contributing

This project is the result of a semester project and is not actively maintained. However, if you find any bugs or have any suggestions for improvement, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.