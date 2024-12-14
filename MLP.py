import random
import numpy as np
import math
def sig(x):
    #Sigmoid Function
    return 1/(1+np.exp(-x))
def sigDeriv(x):
    #Derivative of Sigmoid Function
    return sig(x)*(1-sig(x))
class Node:
    def __init__(self,bias):
        #Creates a node with specified bias
        self.bias = bias
        self.value = 0
        self.incomingEdges = []
        self.outgoingEdges = []
        self.error = 0
        self.input = 0
        self.errorDerivative = 0
    def setValue(self,value):
        #Sets the value of a node
        self.value = value
    def updateValue(self):
        #Updates the value of a node based on the bias and the values of the incoming edges. 
        if self.incomingEdges:
            self.value = self.bias
            for edge in self.incomingEdges:
                self.value = self.value + edge.value
            self.value = sig(self.value)
    def __str__(self):
        return f"{self.bias}"
class Layer:
    def __init__(self,numberOfNodes):
        #Creates a layer with a specified number of nodes
        self.nodes = []
        for i in range(numberOfNodes):
            self.nodes.append(Node(2*random.random()-1))
        self.outgoingEdges = []
    def __str__(self):
        return f"{self.nodes}"
    def setValue(self,valueArray):
        #Sets the value of all of the nodes in a layer. Used to input values into the first layer of the neural network 
        if len(valueArray) != len(self.nodes):
            raise Exception("Wrong number of values")
        for i in range(len(valueArray)):
            self.nodes[i].setValue(valueArray[i])
class Edge:
    def __init__(self,fromNode,toNode,weight):
        #Creates an edge between two nodes with a specified weight
        self.fromNode = fromNode
        self.toNode = toNode
        self.weight = weight
        self.value = 0
        self.errorDerivative = 0
        self.totalErrorDerivative = 0
        toNode.incomingEdges.append(self)
        fromNode.outgoingEdges.append(self)
    def updateValue(self):
        #Updates the value of an edge based on the previous nodes value.
        self.value = self.weight * self.fromNode.value
    def __str__(self):
        return f"{self.weight(self.fromNode)(self.toNode)}"
class MultiLayerPerceptron:
    def __init__(self,LayerArray):
        #Creates a neural network with a specified number of layers and nodes in each layer
        self.LayerCount = len(LayerArray)
        self.Layers = []
        for i in LayerArray:
            self.Layers.append(Layer(i))
        for layernumber in range(self.LayerCount - 1): #
            #Creates edges between each node in each layer
            fromLayer = self.Layers[layernumber]
            toLayer = self.Layers[layernumber + 1]
            for fromNode in fromLayer.nodes:
                for toNode in toLayer.nodes:
                    fromLayer.outgoingEdges.append(Edge(fromNode,toNode,2*random.random()-1))
    def Evaluate(self,inputArray):
        #Calculates the output of the neural network based on some input.
        if len(inputArray) != len(self.Layers[0].nodes):
            raise Exception("Wrong number of Inputs")
        for layer in self.Layers:
            #Sets the value of each node and edge to 0.
            for node in layer.nodes:
                node.value = 0
            for edge in layer.outgoingEdges:
                edge.value = 0 
        firstLayer = self.Layers[0]
        firstLayer.setValue(inputArray)  
        for edge in firstLayer.outgoingEdges:
            #Updates value of edges from first layer
            edge.updateValue()
        for layer in self.Layers[1:]:
            #Updates values of edges and nodes in layers after first layer
            for node in layer.nodes:
                node.updateValue()
            for edge in layer.outgoingEdges:
                edge.updateValue()
        output = []
        for node in self.Layers[-1].nodes:
            #Outputs values of nodes in last layer
            output.append(node.value)
        return output
    def Backpropagate(self,trainingData,learningRate):
        #Calculates the error of nodes in the network and updates weights and biases
        output = self.Evaluate(trainingData[0]) #Runs the network in the forward direction
        lastLayer = self.Layers[-1]
        for i in range(len(lastLayer.nodes)):
            #Calculates error of nodes in the last layer
            lastLayer.nodes[i].errorDerivative = (output[i] - trainingData[1][i])
        for layer in self.Layers[-2::-1]:
            #Calculates error of nodes and edges in all layers except the last
            for node in layer.nodes:
                node.errorDerivative = 0
                for edge in node.outgoingEdges:
                    input = node.value
                    weight = edge.weight
                    toNodeError = edge.toNode.errorDerivative
                    edge.errorDerivative = 0
                    node.errorDerivative = node.errorDerivative + (input * (1 - input) * weight * toNodeError)
                    edge.errorDerivative = input * toNodeError
        for layer in self.Layers:
            #Updates weights and biases
            for node in layer.nodes:
                error = node.errorDerivative
                node.bias = node.bias - (learningRate * error)
            for edge in layer.outgoingEdges:
                error = edge.errorDerivative
                edge.weight = edge.weight - (learningRate * error)
    def Train(self,trainingData,learningRate):
        #Trains the neural network
        for i in trainingData:
            self.Backpropagate(i,learningRate)
    def __str__(self):
        return f"{self.Layers}"

trainingData = []
for i in range(10725):
    input = 5*(random.random()-0.5)
    trainingData.append([[input],[sig(input)]])
neuralNet = MultiLayerPerceptron([1,2,3,3,2,1])
neuralNet.Train(trainingData,0.751)
print(neuralNet.Evaluate([0.7]))
print(sig(0.7))
