#!/usr/bin/env python
"""
DomainTransform.py is a collection of classes for doing domain transforms
"""
__author__ = "Steven Munn"
__email__ = "sjmunn@umail.ucsb.edu"
__date__ = "10-09-2015"
# Dependancies
# Runtime communication
import logging

# Linear Algebra and scientific computing
import numpy as np
from scipy import sparse as sparse
from scipy.sparse import linalg as LA
from scipy import linalg as LAdense

import copy

# NetworkX
import networkx as nx

"""
Function takes the element-wise inverse of a ROW vector
and returns a sparse csr_matrix with diagonal entries
"""
def invertDiagonal(diagMatrix):
    # Check the input matrix is an numpy object
    # assert type(diagMatrix).__module__ == np.__name__
    rowInverse = (1.0/diagMatrix)
    invMatrix = sparse.diags( rowInverse, 0).tocsr()
    return invMatrix

"""
The universe class describes the graph which we are studying
"""

class Universe:
    def matrixRepresentation(self):
        # Laplacian here is the L=D-A definition from Network X
        # Get Laplacian and Adjancency matrix
        self.Sparse_L = nx.laplacian_matrix(self.Graph)
        self.Sparse_A = nx.adjacency_matrix(self.Graph)

        self.L = sparse.csr_matrix(self.Sparse_L)
        self.A = sparse.csr_matrix(self.Sparse_A)

        # Degree Matrix of graph
        self.Drow = sparse.csr_matrix.sum(self.A,axis=1)
        self.Drow = np.array(self.Drow).reshape(-1)
        self.D = sparse.diags(self.Drow, 0).tocsr()

        # Invert the degree matrix
        self.Dinv = invertDiagonal(self.Drow)

        # Probability Transition Matrix
        self.P=sparse.csr_matrix.dot(self.A,self.Dinv)

    def printMatrices(self):
        self.log.debug("Laplacian: ")
        self.log.debug(self.L)
        self.log.debug("Adjacency: ")
        self.log.debug(self.A)

    def shortestPath(self,source,target):
        path=nx.shortest_path(self.Graph,source,target)
        return path

    def betweennessCentrality(self,filename):
        bcFile = filename + "_bc.npy"
        try:
            centralityVector = np.load(bcFile)
        except IOError:
            self.log.debug("Beginning betweenness centrality")
            centralityVector = nx.betweenness_centrality(self.Graph)
            self.log.debug("Done with betweenness centrality")
            centralityVector = np.array(centralityVector.values())
            np.save(bcFile,centralityVector)
        return centralityVector
    
    def pagerank( self):
        prVector = nx.pagerank(self.Graph,alpha=.85)
        prVector = np.array(prVector.values())
        return prVector

    # Add node number as an attribute to the Graph
    def _nodeAnnotate (self):
        for n,d in self.Graph.nodes_iter(data=True):
            self.Graph.node[n]['number'] = n

    def annotateSignal (self,signal,signalName='signal'):
        signalNorm=signal # /float(np.amax(signal))
        for n,d in self.Graph.nodes_iter(data=True):
            #assert len(signalNorm[n]) == 1, \
            #    'Signal Norm is longer than one'
            self.Graph.node[n][signalName] = np.asscalar(signalNorm[n])

    def initRandom(self,innNodes):
        # Construct the graph randomly
        self.nNodes=innNodes
        self.Graph=nx.Graph()
        self.Graph=nx.powerlaw_cluster_graph(self.nNodes,2,.3)
        self._nodeAnnotate()

    def initFromData(self,edges):
        self.log.debug("Graph is in undirected mode.")
        self.log.debug("Check the initFromData function")
        self.log.debug("from UniverseDomains.py to toggle")
        self.log.debug("directionality.")
        self.Graph=nx.Graph()
        self.Graph.add_edges_from(edges)
        self.nNodes=self.Graph.number_of_nodes()
        self._nodeAnnotate()

    def removeNode(self, node):
        self.Graph.remove_node(node)
        self.nNodes=self.Graph.number_of_nodes()

    def __init__(self):
        self.log=logging.getLogger("Random_Walk_Sim")


"""
Domain Transform is the parent class for all the transforms we will define herein
"""
class DomainTransform:
    def transform(self,signal):
        self._checkBasis()
        self.log.debug("Signal")
        self.log.debug(signal)
        fHat = np.dot(np.transpose(self.basis),signal)
        self.log.debug("Transform signal")
        self.log.debug(fHat)
        return fHat

    def reconstruct(self,coefs):
        self._checkBasis()
        signal = np.dot(self.basis,coefs)
        return signal

    def orthogonalityCheck(self):
        self._checkBasis()
        # Multiplies the basis matrices to check they are orthogonal in
        # unit testing
        QQ=np.dot(np.transpose(self.basis),self.basis)
        return np.real(QQ)

    def _checkBasis(self):
        try:
            getattr(self, 'basis')
        except AttributeError:
            raise AttributeError("call buildBasis() method")

    def __init__(self,inHome):
        self.log=logging.getLogger("Random_Walk_Sim")
        self.Home=inHome

        # Domain matrices
        try:
            self.A=inHome.A
        except AttributeError:
            raise AttributeError('Call the Universe matrixRepresentation() method before instantiating a Domain Transform class')

        self.L=inHome.L
        
        # Degree Matrix of graph
        self.D=inHome.D

        # Probability Transition Matrix
        self.P=inHome.P


"""
Normalized Domain
"""
class NormalDomain(DomainTransform):
    def prefixTransform(self,signal):
        signal = sparse.csr_matrix.dot(self.DnegHalf,signal)
        fHat = np.dot(np.transpose(self.basis),signal)
        return fHat

    def prefixReconstruct(self,coefs):
        signal = np.dot(self.basis,coefs)
        signal = sparse.csr_matrix.dot(self.DposHalf,signal)
        return signal

    def findSteadyState(self,filename):
        try:
            self.steadyVect = np.load(filename)
        except IOError:
            self.log.info(" ==== Computing steady-state ==== ")
            eigenVal, eigVect = LA.eigs(self.P, 1)
            self.steadyVect = np.real(eigVect)
            np.save(filename, self.steadyVect)
            self.log.info(" ==== Done Building steady-state  ==== ")

    def topNodeFromSteady(self):
        if hasattr(self, 'steadyVect'):
            return np.argmax(self.steadyVect)
        else:
            raise AttributeError("Compute Steate State first!")

    def buildBasis(self, filename, kVectors=None, which='SM'):
        eigsFile = filename + "_eigvals.npy"
        eigVectFile = filename + "_eigvect.npy"

        try:
            self.basis = np.load(eigVectFile)
            self.lambdas = np.load(eigsFile)
            
            self.log.info(" ==== Loaded Basis from file ==== ")
            
        except IOError:
            if kVectors == None:
                self.kVects = sparse.csr_matrix.get_shape(self.A)[0] - 2

            self.log.info(" ==== Builing Domain Basis ==== ")
            eigVals, eigVects = LA.eigsh(self.Lw,self.kVects,which='SM')
            self.lambdas, self.basis = self._sortEigen(eigVals, eigVects)
            np.save(eigsFile, self.lambdas)
            np.save(eigVectFile, self.basis)
            self.log.info(" ==== Done Building Basis  ==== ")

    # DO NOT CALL THIS WITH LARGE MATRICES
    def buildFullBasis(self,filename):
        self.log.warning(" =!!= Computing the full basis can be slow =!!= ")

        eigsFile = filename + "_MDfull_eigvals.npy"
        eigVectFile = filename + "_MDfull_eigvect.npy"

        try:
            self.basis = np.load(eigVectFile)
            self.lambdas = np.load(eigsFile)
            
            self.log.info(" ==== Loaded Basis from file ==== ")

        except IOError:
            self.log.info(" ==== Builing Domain Basis ==== ")
            eigVals, eigVects = LAdense.eigh(self.Lw.todense())
            self.log.debug("Eigenvectors: ")
            self.log.debug(eigVects)
            self.log.debug("Eigenvalues: ")
            self.log.debug(eigVals)
            self.lambdas, self.basis = self._sortEigen(eigVals, eigVects)
            np.save(eigsFile, self.lambdas)
            np.save(eigVectFile, self.basis)
            self.log.info(" ==== Done Building Basis  ==== ")

    def _sortEigen(self,eigenValues,eigenVectors):
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        return eigenValues, eigenVectors

    def __init__(self,inHome):
        self.log=logging.getLogger("Random_Walk_Sim")
        DomainTransform.__init__(self,inHome)

        # Random Walk Normalized Laplcian
        # L_{sym} = I - D^{-1/2} A D^{-1/2}
        self.DnegHalf = sparse.csc_matrix.sqrt(self.D.tocsc())
        self.DnegHalf = invertDiagonal(self.DnegHalf.diagonal())
        self.DposHalf = sparse.csc_matrix.sqrt(self.D.tocsc())
        self.DposHalf = self.DposHalf.tocsr()

        I=sparse.eye(self.D.shape[0]).tocsr()
        self.N = sparse.csr_matrix.dot(self.DnegHalf,sparse.csr_matrix.dot(self.A,self.DnegHalf))
        self.Lw=I-sparse.csr_matrix.dot(self.DnegHalf,sparse.csr_matrix.dot(self.A,self.DnegHalf))

"""
Un-normalized Laplacian Domain
"""
class LaplacianDomain(DomainTransform):
    def buildBasis(self, filename, kVectors=None, which='SM'):
        eigsFile = filename + "_Lap_eigvals.npy"
        eigVectFile = filename + "_Lap_eigvect.npy"
        
        try:
            self.basis = np.load(eigVectFile)
            self.lambdas = np.load(eigsFile)

            self.log.info(" ==== Loaded Basis from file ==== ")
            
        except IOError:
            if kVectors == None:
                self.kVects = sparse.csr_matrix.get_shape(self.A)[0] - 2

                self.log.info(" ==== Builing Domain Basis ==== ")
                eigVals, eigVects = LA.eigsh( self.L.asfptype(), self.kVects, which='SM')
                self.lambdas, self.basis = self._sortEigen( eigVals, eigVects)
                np.save(eigsFile, self.lambdas)
                np.save(eigVectFile, self.basis)
                self.log.info(" ==== Done Building Basis  ==== ")

    def buildFullBasis(self,filename):
        self.log.warning(" =!!= Computing the full basis can be slow =!!= ")

        eigsFile = filename + "_LapFull_eigvals.npy"
        eigVectFile = filename + "_Lapfull_eigvect.npy"

        try:
            self.basis = np.load(eigVectFile)
            self.lambdas = np.load(eigsFile)
            
            self.log.info(" ==== Loaded Basis from file ==== ")

        except IOError:
            self.log.info(" ==== Builing Domain Basis ==== ")
            eigVals, eigVects = LAdense.eigh(self.L.todense())
            self.lambdas,self.basis=self._sortEigen(eigVals,eigVects)
            np.save(eigsFile, self.lambdas)
            np.save(eigVectFile, self.basis)
            self.log.info(" ==== Done Building Basis  ==== ")

    def _sortEigen(self,eigenValues,eigenVectors):
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        return eigenValues, eigenVectors

    def __init__(self,inHome):
        self.log=logging.getLogger("Random_Walk_Sim")
        DomainTransform.__init__(self,inHome)
        # No caluclations needed as L is already given by the
        # Universe class

"""
G Domain
"""

class gDomain(DomainTransform):
    def prefixTransform(self,signal):
        fHat = np.dot(self.inverseBasis,signal)
        return fHat

    def prefixReconstruct(self,coefs):
        signal = np.real(np.dot(self.basis,coefs))
        return signal
    
    def buildFullBasis(self,filename):
        self.log.warning(" =!!= Computing the full basis can be slow =!!= ")

        eigsFile = filename + "_gFull_eigvals.npy"
        eigVectFile = filename + "_gfull_eigvect.npy"
        eigVectInv = filename + "_gfull_eigvectInv.npy"

        try:
            self.basis = np.load(eigVectFile)
            self.lambdas = np.load(eigsFile)
            self.inverseBasis = np.load(eigVectInv)
            
            self.log.info(" ==== Loaded Basis from file ==== ")

        except IOError:
            self.log.info(" ==== Builing Domain Basis ==== ")
            eigVals, eigVects = LAdense.eig(self.gMat)
            self.lambdas,self.basis=self._sortEigen(eigVals,eigVects)
            self.inverseBasis = LAdense.inv(self.basis)
            self.basis = self.basis
            self.lambdas = self.lambdas
            self.inverseBasis = self.inverseBasis
            np.save(eigsFile, self.lambdas)
            np.save(eigVectFile, self.basis)
            np.save(eigVectInv, self.inverseBasis)
            self.log.info(" ==== Done Building Basis  ==== ")

    def _sortEigen(self,eigenValues,eigenVectors):
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        return eigenValues, eigenVectors

    def __init__(self,inHome):
        self.log=logging.getLogger("Random_Walk_Sim")
        DomainTransform.__init__(self,inHome)
        self.gMat = nx.google_matrix(inHome.Graph,alpha=.85)
        self.gMat = self.gMat.transpose()
        # No caluclations needed as L is already given by the
        # Universe class


"""
Overcomplete gDomain A
"""

class gDomainA:
    def transform(self,signal):
        signal = self.normalizePin(signal)
        self.computeDelP(signal)
        fHat = np.dot(self.basisInv,self.delP)
        return fHat

    def reconstruct(self,coefs):
        signal = np.dot(self.basis,coefs)
        signal = self.normConstant * (signal + self.pageRank)
        return signal
    
    def getPR(self):
        self.pageRank = self.Home.pagerank()

    def normalizePin(self,pIn):
        self.normConstant = pIn.sum()
        pIn = pIn / float(self.normConstant)
        return pIn

    # ==================== PROBLEM WITH pIn
    def computeDelP(self,pIn):
        self.log.debug("Shape for pIn[:]")
        # self.log.debug(np.shape(pIn[:,0]))
        self.log.debug("Shape for self.pageRank")
        self.log.debug(type(self.pageRank))
        
        self.delP = pIn[:,0] - self.pageRank
        # self.delP = pIn - self.pageRank
        self.log.debug("Shape for delP")
        self.log.debug(np.shape(self.delP))
        

    
    def buildProjectionBasis(self,filename):
        self.log.info("Building the projection basis")
        projectionBasisFile = filename + "_gDomainA_proj_basis.npy"

        try:
            self.PB = np.load(projectionBasisFile)
        except IOError:
            PB = np.zeros([self.nNodes, self.nNodes])
            for i in range(0,self.nNodes):
                negativeVector = copy.copy(self.pageRank)
                negativeVector[i] = 0
                # negativeVector = negativeVector / (1-self.pageRank[i])
                negativeVector = 1 - negativeVector / (1 - self.pageRank[i])
                # negativeVector = negativeVector / self.nNodes
                PB[:,i] = - negativeVector
                PB[i,i] = 1
            
            self.PB = PB
            np.save(projectionBasisFile,PB)

        self.log.info("Done with projection basis")

    def project(self):
        try:
            self.projectionCoefs = np.dot(self.PB,self.delP)
        except AttributeError:
            raise "Call buildProjectionBasis() before project()"

        return self.projectionCoefs

    def findMinimumCoefficient(self):
        try:
            self.minIdx = np.argpartition(np.abs(self.projectionCoefs),1,0)[:1]
        except IOError:
            raise "Call project() before findMinimumCoefficient"
        self.log.debug("minIdx is:")
        self.log.debug(self.minIdx)
            
    def buildDomain(self, filename):
        self.log.info("Building the final domain")
        basisFile = filename + "_gDomainA_basis.npy"
        basisInvFile = filename + "_gDomainA_basisInv.npy"

        try:
            self.basis = np.load(basisFile)
            self.basisInv = np.load(basisInvFile)
        except IOError:
            # use copy.copy if need to keep PB
            self.basis = self.PB
            self.basis[:,self.minIdx] = np.ones([self.nNodes,1])
            self.basisInv = LAdense.inv(self.basis)
            np.save(basisFile,self.basis)
            np.save(basisInvFile,self.basisInv)

        self.log.info("Done building the domain")
    def __init__(self,inHome):
        self.log=logging.getLogger("Random_Walk_Sim")
        self.Home = inHome
        self.nNodes = self.Home.Graph.number_of_nodes()

        
"""
Overcomplete Betweenness centrality Domain A
"""

class BeDomainA:
    def transform(self,signal):
        signal = self.normalizePin(signal)
        self.computeDelP(signal)
        fHat = np.dot(self.basisInv,self.delP)
        return fHat

    def reconstruct(self,coefs):
        signal = np.dot(self.basis,coefs)
        signal = self.normConstant * (signal + self.betweenSignal)
        return signal
    
    def getBC(self,filename):
        self.betweenC = self.Home.betweennessCentrality(filename)
        N = self.nNodes
        self.betweenSignal = (N-2)*(N-1)*self.betweenC + N-1
        self.betweenSignal = self.betweenSignal / np.sum(self.betweenSignal)

    def normalizePin(self,pIn):
        self.normConstant = pIn.sum()
        pIn = pIn / float(self.normConstant)
        return pIn

    def computeDelP(self,pIn):
        self.log.debug("Shape for pIn[:]")
        self.log.debug(np.shape(pIn[:,0]))
        self.log.debug("Shape for self.betweenSignal")
        self.log.debug(type(self.betweenSignal))
        
        self.delP = pIn[:,0] - self.betweenSignal
        self.log.debug("Shape for delP")
        self.log.debug(np.shape(self.delP))

    def buildProjectionBasis(self,filename):
        self.log.info("Building the projection basis")
        projectionBasisFile = filename + "_gDomainA_proj_basis.npy"

        try:
            self.PB = np.load(projectionBasisFile)
        except IOError:
            PB = np.zeros([self.nNodes, self.nNodes])
            for i in range(0,self.nNodes):
                negativeVector = copy.copy(self.betweenSignal)
                negativeVector[i] = 0
                negativeVector = negativeVector / (1-self.betweenSignal[i])
                # negativeVector = negativeVector / (self.nNodes)
                PB[:,i] = - negativeVector
                PB[i,i] = 1
            
            self.PB = PB
            np.save(projectionBasisFile,PB)

        self.log.info("Done with projection basis")

    def project(self):
        try:
            self.projectionCoefs = np.dot(self.PB,self.delP)
        except AttributeError:
            raise "Call buildProjectionBasis() before project()"

        return self.projectionCoefs

    def findMinimumCoefficient(self):
        try:
            self.minIdx = np.argpartition(np.abs(self.projectionCoefs),1,0)[:1]
        except IOError:
            raise "Call project() before findMinimumCoefficient"
        self.log.debug("minIdx is:")
        self.log.debug(self.minIdx)
            
    def buildDomain(self, filename):
        self.log.info("Building the final domain")
        basisFile = filename + "_BeDomainA_basis.npy"
        basisInvFile = filename + "_BeDomainA_basisInv.npy"

        try:
            self.basis = np.load(basisFile)
            self.basisInv = np.load(basisInvFile)
        except IOError:
            # use copy.copy if need to keep PB
            self.basis = self.PB
            self.basis[:,self.minIdx] = np.ones([self.nNodes,1])
            self.basisInv = LAdense.inv(self.basis)
            np.save(basisFile,self.basis)
            np.save(basisInvFile,self.basisInv)

        self.log.info("Done building the domain")
    def __init__(self,inHome):
        self.log=logging.getLogger("Random_Walk_Sim")
        self.Home = inHome
        self.nNodes = self.Home.Graph.number_of_nodes()

"""
Build Haar wavelts from normalized cuts
"""

class GraphHaar:
    def cutNormalized(self, subA, haarPosition):
        
        # Degree Matrix of graph
        subDrow = sparse.csr_matrix.sum(subA,axis=1)
        Drow = np.array(subDrow).reshape(-1)
        subD = sparse.diags(self.Drow, 0).tocsr()

        # D^(-1/2)
        subDnegHalf = sparse.csc_matrix.sqrt(subD.tocsc())
        subDnegHalf = invertDiagonal(subDnegHalf.diagonal())

        # A_{n} = D^{-1/2} A D^{-1/2}
        subAn = sparse.csr_matrix.dot( subDnegHalf, sparse.csr_matrix.dot( subA, subDnegHalf))

        eigVals, eigVects = LA.eigsh(subAn, 1, which='SM')
        
        
    def haarBuilder(self):
        A = self.Home.A
        self.basis = np.zeros([self.nNodes, self.nNodes])
        self.basis[0] = np.ones(self.nNodes)
        
    
    def buildDomain(self, filename):
        self.log.info("Building the Haar wavelets")
        basisFile = filename + "_GraphHaar_Basis.npy"

        try:
            self.basis = np.load(basisFile)
        except IOError:
            self.haarBuilder()
            np.save(basisFile,self.basis)

        self.log.info("Done building the domain")
    
    def __init__(self,inHome):
        self.log = logging.getLogger("Random_Walk_Sim")
        self.Home = inHome
        self.nNodes = self.Home.Graph.number_of_nodes()
