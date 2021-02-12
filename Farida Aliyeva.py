
import numpy as np
import pandas as pd
import heapq
import math
import time
import argparse 

# Function to read the source and destination Files. Then Strings are split and numbers retrieved
def readSD(filename):
  with open(filename,'r') as sd:
    for line in sd.readlines():
      if line.startswith("S"):
        Source = int(line.split(',')[1]) 
      elif line.startswith("D"):
        Destination = int(line.split(',')[1]) 
  return Source,Destination

# Function to read the vertices File. Appending the normalizaed valued of square ids from file to grid data structure
def readVertices(n,filename):
  grid = []
  for i in range(n):
    grid.append(0)
  with open(filename,'r') as vertices:
    for line in vertices.readlines():
      if not line.startswith("#"):
        vertex,squareId = line.split(",")
        vertex,squareId = int(vertex),int(squareId)
        grid[vertex] = [squareId/10,squareId%10]
    return grid

# Function to calculate Euclidian distance used primarily for uninformed search in our case A*. Adds all the distances to the data structure.
def Euclidian(Dest,n,vertices):
  distances=[]
  for i in range(n):
    x = (abs(vertices[i][0] - vertices[Dest][0]))
    y = (abs(vertices[i][1] - vertices[Dest][1]))
 
    distance = math.sqrt(math.pow(x,2) +math.pow(y,2))*100
    distances.append(distance)
  return distances
#Function to read the Edge values from the file. The same way as vertices were extracted. Storing the values at edgesValues data structure.
def readEdges(filename):
  edgesValues=[]
  with open(filename,'r') as edges:
    for line in edges.readlines():
      if not line.startswith("#"):
        edgesValues.append(line.split(','))
  return edgesValues

# This function created a grid from the input data (edges) to build a graph for our data. Which in turn stores the nodes and values of heuristics.
def create_grid(edg,heuristic):  
  edge_grid=[]
  for i in range(100):
    edge_grid.append([])
  for line in edg:
    vert_from,vert_to,d = int(line[0]),int(line[1]),int(line[2])
    edge_grid[vert_from].append([d+heuristic[vert_to],vert_to,d])
    edge_grid[vert_to].append([d+heuristic[vert_from],vert_from,d])
  return edge_grid

#Function used by Astar function to print out the resulting path
def print_path_Astar(path,parent,D,S):
  path.append(parent[D])
  if parent[D] != 0:
    print_path_Astar(path,parent,parent[S],0)
    #descending for loop.
  for i in range(len(path) -1, -1,-1):
    print(path[i],end=" ")
#AStar Algorithm path search algorithm searches for the shortest path between Starting and Final states
#Traversing the nodes we add the very first one to the status data structures comparing the costs  which are calculated by (g(n)+ h(n)).
#Once done the node is moved to visited list. Storing the parent nodes we later on print out the Path itself for the Algorithm.
#Reaching Destination node we end the search and come up with the shortest path for the algorithm. In case if the Destination node can not
#be found the Path from Source to Destination can not be estimated then. Throughout the searching process the live score of the process is calculated.
# HeapQ is known as priority queue in python data structures.
def Astar(n,Source,Destination, grid_edges_h):
  start = time.time()
  cost = Source
  parent = []
  status = []
  visited = []
  path = [Destination]
  for i in range(n):
    visited.append(0)
    parent.append(0)
  heapq.heappush(status, (cost, [Source,cost,Source]))
  while len(status) != 0:
    live_score,[live_node,live_cost,p] = heapq.heappop(status)
    if visited[live_node] == 0:
      visited[live_node] = 1
      parent[p] = p
      if live_node == Destination:
        finish = time.time()
        print(f"Cost of Shortest Path = {int(live_score)}")    
        print(f"\nTime: {finish - start} seconds")       
        print('\n Path:' )
        print_path_Astar(path,parent,Destination,Source)
      else:
        for x in grid_edges_h[live_node]:
          heapq.heappush(status, (x[0] + live_cost, [x[1],x[2]+live_cost,live_node]))
#Dikstra algorithm another way to find the shortest path bu comparing to A star it 
#does not thrive to find a better path using the heuristic funciton. The main objective of Dijkstra is just to 
#explore all the possible paths. the main objective is to provide a function with the infinite value and a min value
# and use them to compare and find the minimum which is in fact a shortest path. 
def Dijkstra(n,edg,Source,Destination):
    def printPath(num):
      if (num == -1):
          return
      printPath(parent[num])
      print(num, end=" ")
# a way to find a short path 
    def stop(a, b):
        if (dist[a] + h[a][b] < dist[b]):
            dist[b] = dist[a]+h[a][b]
            parent[b] = a
    infinite = 1000000000
    visited = np.zeros(n)
    h = np.full((n,n), infinite)
    start = time.time()
    for line in edg:
        v_from,v_to,len = int(line[0]), int(line[1]),int(line[2])
        h[v_from][v_to] = h[v_to][v_from] = len

    dist = np.full(n, infinite)
    dist[Source]=0
    parent = np.full(n, -1)

    for i in range(0, n-1):
        min = infinite
        v = -1
        for j in range (0, n):
            if (visited[j]==0 and dist[j] <min):
                min = dist[j]
                v = j
                if (v==-1):
                    break
                for j in range(0, n):
                    if (visited[j]==0 and h[v][j]!=infinite):
                        stop(v, j)
                        visited[v]=1

    if (dist[Destination]==infinite):
        print("-1")
    else:
        print(f"Cost of Shortest Path = {dist[Destination]}")   
        finish = time.time()
        print(f"\nTime: {finish - start} seconds" )          
        print('\n Path:' )
        printPath(Destination)
#Run program method which takes number of edges, the algorithm type and file names from the user
def run_program(num_of_edges,algorithm,vertexfile, edgefile, sdfile):
   Source,Destination = readSD(sdfile)
   vertices = readVertices(num_of_edges,vertexfile)
   edg = readEdges(edgefile)
   heuristic = Euclidian(Destination, num_of_edges,vertices)
   grid_edges_h = create_grid(edg,heuristic)
   if(algorithm == 'Astar'):
     Astar(num_of_edges,Source,Destination,grid_edges_h)
   elif(algorithm=='Dijkstra'):
      Dijkstra(num_of_edges,edg, Source, Destination)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('nodes',type =int)
    parser.add_argument('algorithm',type=str)
    parser.add_argument('vertexfile',type=str)
    parser.add_argument('edgefile',type =str)
    parser.add_argument('sdfile',type=str)
    args = parser.parse_args()

    nodes = args.nodes
    algorithm = args.algorithm
    vertexfile= args.vertexfile
    edgefile= args.edgefile
    sdfile= args.sdfile

    run_program(nodes,algorithm,vertexfile,edgefile,sdfile)

#uncomment this part to see the results in google colab or jupyter notebook
#run_program(100,'Astar','Vertices.txt','edges.txt','source and destination.txt')

#uncomment this part to see the results in google colab or jupyter notebook
#run_program(100,'Dijkstra','Vertices.txt','edges.txt','source and destination.txt')

