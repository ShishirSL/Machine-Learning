# Authors:

# Shishir Singapura Lakshminarayan - ssl495
# Mayank Grover	- mg5229



Instructions on how to run the program:


* Keeping the main program `IncomePrediction.py` in the same folder as the data files, use the following format/syntax


* $ python IncomePrediction.py <train_file_name.csv> <test_file_name.csv>


The program will generate the prediction file `test_outputs.csv` in the same folder as the program.




Requirements for running the program (python modules):


* pandas
* numpy
* sklearn.tree.DecisionTreeRegressor
* sklearn.ensemble.RandomForestRegressor
* sklearn.model_selection.cross_validate
* sklearn.feature_selection.RFE