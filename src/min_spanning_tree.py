#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 10:45:29 2021

@author: romainloirs
"""

#https://www.programiz.com/dsa/kruskal-algorithm


# Kruskal's algorithm in Python


class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    # Search function

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def apply_union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    #  Applying Kruskal algorithm
    def kruskal_algo(self):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.apply_union(parent, rank, x, y)
        res=0
        for u, v, weight in result:
            #print("%d - %d: %d" % (u, v, weight))
            res+=weight
        return(res)


def kruskal_algo(matrice_adj):
    n=matrice_adj.shape[0]
    g = Graph(n)
    for i in range(n):
        for j in range(n):
            if matrice_adj[i][j]!= 0 or matrice_adj[i][j]!= 10000:
                g.add_edge(i, j, matrice_adj[i][j])
                
    return(g.kruskal_algo())
