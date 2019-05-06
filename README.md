# Pure Python/Numpy Implementations of Simple NN Modules
- Most of the useful implementations is in `layers.py`, `optimizers.py`, and `initializers.py`.


# (5329 only) Instruction for Training the Model, and saving the predictions

1. First, goto `config.py` in the `Algorithm` folder,
    - change the `DATA_PATH` variable to the path that contains the training data and testing inputs (i.e. `train_128.h5`, `train_label.h5`, `test_128.h5`)
    - change the `SAVE_PATH` variable to the path to save `Predicted_labels.h5`
    - change the `PREDICTING` variable to True

2. Then, simply run `python3 train.py` from the `Algorithm` folder
    - the software would load the training data
    - start training on the training configurations specified in `config.py`

3. After training is done, the `Predicted_labels.h5` file will be saved to the `SAVE_PATH` in `config.py`

4. Enjoy!

