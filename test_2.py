import networkx as nx
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import csv

# 中文输出
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

# 度中心性
def centralityDegree(G):
    centrality = {}
    s = 1.0 / (len(G) - 1.0)
    centrality = {n : d * s for n, d in G.degree()}
    centrality = sorted(centrality.items(), key = lambda item:item[1],reverse = True)
    return centrality

# 特征向量中心性
def centralityEigenvector(G, max_iter=50, tol=0):
    M = nx.to_scipy_sparse_matrix(G, nodelist = list(G), weight = None,dtype=float)
    eigenvalue, eigenvector = sp.linalg.eigs(M.T, k=1, which='LR',maxiter=max_iter, tol=tol)
    largest = eigenvector.flatten().real
    norm = np.sign(largest.sum()) * np.linalg.norm(largest)
    return dict(zip(G, largest / norm))

# katz中心性
def centralityKatz(G, alpha = 0.1, beta = 1.0):
    try:
        nodelist = beta.keys()
        b = np.array(list(beta.values(), dtype = float))

    except AttributeError:
        nodelist = list(G)
        b = np.ones((len(nodelist), 1)) * float(beta)

    A = nx.adj_matrix(G, nodelist=nodelist, weight= None).todense().T
    n = A.shape[0]
    centrality = np.linalg.solve(np.eye(n, n) - (alpha * A), b)
    norm = np.sign(sum(centrality)) * np.linalg.norm(centrality)
    centrality = dict(zip(nodelist, map(float, centrality / norm)))
    return centrality
# PageRank
def Matrix(G, alpha=0.85, personalization=None, nodelist=None,weight='weight'):
    if personalization is None:
        nodelist=G.nodes()
    else:
        nodelist=personalization.keys()
    M=nx.to_numpy_matrix(G,nodelist=nodelist,weight=weight)
    (n,m)=M.shape
    if n == 0:
        return M
    dangling=np.where(M.sum(axis=1)==0)
    for d in dangling[0]:
        M[d]=1.0/n
    M=M/M.sum(axis=1)
    e=np.ones((n))
    if personalization is not None:
            v=np.array(list(personalization.values()),dtype=float)
    else:
        v=e
    v=v/v.sum()
    P=alpha*M+(1-alpha)*np.outer(e,v)
    return P
def PageRank(G, alpha=0.85, personalization=None, weight='weight'):
    if personalization is None:
        nodelist=G.nodes()
    else:
        nodelist=personalization.keys()
    M = Matrix(G, alpha, personalization=personalization,nodelist=nodelist, weight=weight)
    eigenvalues,eigenvectors=np.linalg.eig(M.T)
    ind=eigenvalues.argsort()
    largest=np.array(eigenvectors[:,ind[-1]]).flatten().real
    norm=float(largest.sum())
    centrality=dict(zip(nodelist,map(float,largest/norm)))
    return centrality
# betweenness
def centralityBetweenness(G, k=None, normalized=True):
    betweenness = dict.fromkeys(G, 0.0)
    nodes = G
    for s in nodes:
        S, P, sigma = _single_source_shortest_path(G, s)
        betweenness = _accumulate(betweenness, S, P, sigma, s)
    betweenness = _rescale(betweenness, len(G), normalized=normalized)
    return betweenness
def _single_source_shortest_path(G, s):
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0) # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    D[s] = 0
    Q = [s]
    while Q: # use BFS to find shortest paths
        v = Q.pop(0)
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1: # this is a shortest path, count paths
                sigma[w] += sigmav
                P[w].append(v) # predecessors
    return S, P, sigma
def _accumulate(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1.0 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return betweenness
def _rescale(betweenness, n, normalized, directed=False):
    if normalized:
        if n <= 2:
            scale = None # no normalization b=0 for all nodes
        else:
            scale = 1.0 / ((n - 1) * (n - 2))
    else:
        scale = 0.5

    if scale is not None:
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness
# closeness
def closenessCentrality(G):
    path_length = nx.single_source_shortest_path_length
    nodes = G.nodes()
    closeness_centrality = {}
    for n in nodes:
        sp = path_length(G, n)
        totsp = sum(sp.values())
        if totsp > 0.0 and len(G) > 1:
            closeness_centrality[n] = (len(sp)-1.0) / totsp
            # normalize to number of nodes-1 in connected part
            s = (len(sp)-1.0) / (len(G) - 1)
            closeness_centrality[n] *= s
        else:
            closeness_centrality[n] = 0.0
    return closeness_centrality

# 读取边的文件
edges = open("data_t/edge.csv", "r")
edgeReader = csv.reader(edges)
# 将边加入图中
G = nx.DiGraph()
for item in edgeReader:
    G.add_edge(item[0],item[1])
edges.close()

# 测度
print("度中心性：",)
print(centralityDegree(G))


print("特征向量中心性：")
print(centralityEigenvector(G))

print("Katz中心性：")
print(centralityKatz(G))

print("PageRank：")
print(PageRank(G))

print("Betweenness：")
print(centralityBetweenness(G))

print("Closeness：")
print(closenessCentrality(G))

nx.draw(G,node_color='y',with_labels=True,node_size=300,width=1)

plt.show()

