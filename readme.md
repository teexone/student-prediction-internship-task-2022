# General Idea

The general idea is just simple. Analyzing previous purchases, I have built a correlation matrix. Then, I enhanced it with genres correlation by elevating correlation coefficients for the games of the same genre. 

As a vector of games IDs received, the columns corresponding to these games are multiplied by matrix yielding correlation vectors. Games that appeared to have a maximum correlation coefficient with the given set of games are displayed as recommended ones.

# Improvements 

Since the purchases are quite small (from 4 to 7 games per user), the recommendation list can be improved by including some top purchased games. Furthermore, each game is only related with one genre, while actually it can have connections to many of them, and genres of some games are too much specific (for example, *Oxygen Not Included* has genre Colony Simulator). All these exacerbate correlation searching.

# Structure

I mainly worked with Jupiter notebooks, so I encourage to examine them firstly. I splitted the process into three steps: data load (ETL), data process (EDA) and prediction. 

All the code is copied into `main.py` script with several examples printed on launch

# Code launch

Before the code is launched you have to install `pandas` and `numpy` libraries

```pip install pandas numpy```

To launch the script use the following command (requires Python3.10 installed):

```python main.py```

or

```python3 main.py```