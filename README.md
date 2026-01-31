# lopper

The purpose of this repo is to provide a user interface for pruning the output of a fit scikit-learn DecisionTreeRegressor. 

## Dev Specs

### User Flow

* Specify the path to a pickled fit decision tree file (which was generated previously via joblib dump/load) in a config.yml file
* Run a python script which opens a browser interface showing the decision tree with all its nodes as boxes (binary tree with root on top, specific statistics in each node box, and arrows between nodes)
* Drag and rearrange nodes in the interface (for visual aid or in case they are drawn on top of one another)
* Click on nodes to make the child nodes collapse (recursively, i.e. the children‘s children are also collapsed)
* Click on nodes with collapsed child nodes to expand (only the direct) child nodes
* Once the tree is sufficiently pruned, write a name into a text field and click a button which exports the updated pruned tree into the same pickle format, saves an image of the tree as it appears in the browser at that time (effectively a screenshot), and also saves an Excel file with a row for each node and columns for a path description to the node, the node split feature and split value, and columns for each of the stats in the node boxes

### Additional Requirements and Features

* The program should not close upon exporting the tree as the user could want to save multiple versions from the same unpruned tree. The program will be exited by the user manually
* The project should be 100% Python (with exceptions for HTML / CSS if needed)
* The project should use a pyproject.toml which can be used via uv sync
* The stats in each node box are given by tree.node_stats which is an array of dictionaries of stats to write into the box of each node
* The nodes of the tree should be color coded via a heatmap from lowest to highest value of a specific key in the stats dict (the key is decided by the user in the config.yml). Use a heatmap which is white for values close to the average and two different colors in the extremes
* The repo should contain a Python file which generates an example dataset, fits a decision tree on it, and generates the pickled file that can be used to test the repo functionality (so also including some statistics for the node boxes). Important: Have a function which takes as inputs a fit tree, the underlying pandas dataframe, and an output path and produces as output the tree as a pickle with added leaf statistics (this function i want to copy out to another project later)

