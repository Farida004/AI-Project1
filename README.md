# AI Project 1 - Shortest Path (Uninformed + Informed Search)
A random graph is given where the “nodes” are located on a 10x10 2D “chess” board, consisting of 100 “squares”. Each square is of size 10 by 10 units.
Additinally we are given input files: vertices, edges start and finish points. The expectations are: to implement a program that takes inputs and calculates the shortest path.
To calculate the shortest path an uninformed search (in my case Dijkstra algorithm) and informed search (A*) should be implemented. Finally the performance of 2 should be compared. 

## Implementation
The following program is written in Python programming language. The reasons behind this choice were **faster execution** ,** easier file read process**
Additionally, several python libraries were used such as:

- argparse
- numpy
- heapq
- time
- math

## Requirements
- [Python](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installing/)
- [A*](https://en.wikipedia.org/wiki/A*_search_algorithm)
- [Dijkstra](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)


## How to run
First of all make sure you have Python installed on your computer. Then access the folder where you store both **.py** and **txt** files from cmd. Next step you have to run the .py file and write down the inputs accordingly:

```bash
Farida Aliyeva.py 100 'Dijkstra' 'Vertices.txt 'edges.txt' 'source and destination.txt'
```
