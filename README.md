# Q-Table-Learning
Training an MDP agent using Q-Learning with a Q-table on Molecule Dataset.

## Setup and Initialization

Use the same setup and initialization as found in `active-acquisition` repository. Guide to setup and initialization found [here](https://github.com/leungkean/active-acquisition/blob/a3db4d7525b735386e7bb21c67e1dc41671acf96/afa_guide.txt#L6)

## Train and Evaluate

The `Q-table-replay.py` trains the MDP agent using Q-Learning with a Q-table, and evaluates the agent. 
Here the default dataset is `molecule_20` which includes only the top 20 features from the original `molecule` dataset 
determined using nested cross-validation and a DNN. To train and evaluate an MDP agent using Q-Learning with a Q-table run the following command:

```
python Q-table-replay.py
```

If you want to change any of the hyperparameters such as acquisition cost, learning rate, epsilon... use the `-h` flag for more options.
