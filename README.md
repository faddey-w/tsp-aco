# What it does

 - loads data from database of cities
 - builds matrix of distances between first N cities of selected country
 - optionally can save the matrix to a file in CSV, numpy or readable formats
 - then runs ant-colony optimization algorithm to solve travelling salesman problems on selected cities
 - saves visualization of solution progress to files
 - visualization can be viewed in browser
 
 
# How to use

Run solver:

```bash
python main.py -n $NUMBER_OF_CITIES -C $COUNTRY_CODE
```

Extra options:

```bash
python main.py -h
```

Show visualization:  
put the following address to your browser address line:
`file:///path/to/local/reposity/visualizer.html`

Visualization shows paths between cities as edges. Red edge form the optimal
path in the graph. Brightness of edges means amount of ant's pheromone on this path.  
You can go forth and back between iterations by buttons.
