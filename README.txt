Code usage notes for Randomized Optimization Assignment

To get the code:
git clone https://github.com/bentito/cs7641.git

To recreate the Python environment:
cd cs7641
python -m pip install -r requirements.txt

run:
Assignment2CS7641.py

NB: There are flags that are settable in the source to choose:

    Set:
        NN True for the neural net weight choosing problem & False for the randomized optimization experiments
    Set:
        GRID_SEARCH_<ALG> = True to output a csv file to look for best parameters in a spreadsheet
        PLOT_SPECIALS_<ALG> = True to plot interesting cases
        RUN_SPECIALS_<ALG> = True to run stochastic analysis in some interesting cases
    All flag variables are in ALL_CAPS