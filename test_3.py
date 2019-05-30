import networkx as nx
import matplotlib.pyplot as plt
num=3
G = nx.complete_graph(num)
nx.draw(G,node_size=10,width=0.1)
plt.savefig("8nodes.png")
plt.show()