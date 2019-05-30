import networkx as nx
import matplotlib.pyplot as plt

import csv
# 读取节点和边的文件

nodes = open("data_t/node.csv", "r")
edges = open("data_t/edge.csv", "r")
# nodes = open("data_1/nodes.csv", "r")
# edges = open("data/1-edges.csv", "r")
nodeReader = csv.reader(nodes)
edgeReader = csv.reader(edges)

G = nx.DiGraph()
for item in nodeReader:
    G.add_node(item[0])
for item in edgeReader:
    G.add_edge(item[0],item[1])

nodes.close()
edges.close()


# G.add_node(1)
# G.add_node(2)
# G.add_nodes_from([3,4,5,6])
# G.add_cycle([1,2,3,4])
# G.add_edge(1,3)
# G.add_edges_from([(3,5),(3,6),(6,7)])
nx.draw(G,node_color='y',with_labels=True,node_size=50,font_size=8)
plt.savefig("graph.png")
plt.show()