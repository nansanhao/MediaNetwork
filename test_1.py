import networkx as nx
import matplotlib.pyplot as plt
import csv

# 读取边的文件
edges = open("data_2/edge.csv", "r")
edgeReader = csv.reader(edges)

G = nx.DiGraph()
for item in edgeReader:
    G.add_edge(item[0],item[1])
edges.close()

nx.draw(G,node_color='y',node_size=10,width=0.1)
plt.savefig("graph.png")
plt.show()
