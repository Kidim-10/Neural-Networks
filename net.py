#!/usr/bin/env python3
#import json, codecs
import numpy as np2
import numpy.random as np
from  scipy.special import expit
from math import *
import matplotlib.pyplot as plt
import time

class Net(object):

  #initilize the net
  def __init__(self, inputNodes, hiddenNodes, outputNodes, learning_rate = None):
    #the nodes, input, hidden and output layer
    self.inputNodes = inputNodes
    self.hiddenNodes = hiddenNodes
    self.outputNodes = outputNodes
    
    #initilize the random values of the weights
    #wih the weights between the input and hidden nodes (layer)
    #who the weights between the hidden and output nodes (layer)
    self.wih = np.normal(0.0, pow(self.inputNodes, -0.5), (self.inputNodes, self.hiddenNodes))
    self.who = np.normal(0.0, pow(self.hiddenNodes, -0.5), (self.hiddenNodes, self.outputNodes))
    
    #for the bias
    self.l_rate = learning_rate

    #the activation function
    self.activation_function = lambda x: expit(x)


  #training for the neural network
  def train(self, inputNodes, target_values):
      # convert inputs list to 2d array
      inputs = np2.array(inputNodes, ndmin=2).T
      targets = np2.array(target_values, ndmin=2).T
      
      # calculate signals into hidden layer
      hidden_inputs = np2.dot(self.wih.T, inputs)
      # calculate the signals emerging from hidden layer
      hidden_outputs = self.activation_function(hidden_inputs)

      # calculate signals into final output layer
      final_inputs = np2.dot(self.who.T, hidden_outputs)
      # calculate the signals emerging from final output layer
      final_outputs = self.activation_function(final_inputs)
      
      # output layer error is the (target - actual)
      output_errors = targets - final_outputs
      # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
      hidden_errors = np2.dot(self.who, output_errors)

      #the problem is by self.who +=....
      a = (output_errors * final_outputs * (1.0 - final_outputs))
      b = np2.dot(a, np2.transpose(hidden_outputs))
      np2.array(self.l_rate * b)
      final_learn_rate = self.l_rate * b
      self.who += final_learn_rate.T
      # update the weights for the links between the hidden and output layers
      #self.who += self.l_rate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
     
      #the second wih update after the error detection 
      output_derivative = (hidden_errors * hidden_outputs * (1.0 - hidden_outputs))
      product = np2.dot(output_derivative, np2.transpose(inputs))
      final_learn_rate2 = self.l_rate * product 
      self.wih += final_learn_rate2.T 
      
      f = np2.array(self.wih)
      print(np2.shape(f))
      plt.plot(f)
      plt.show(block = False)
      plt.pause(0.01)
      plt.close() 

      pass

   #query the neural network
  def query(self,inputs_list):
      # convert inputs list to 2d array
      inputs = np2.array(inputs_list, ndmin=2).T
      
      # calculate signals into hidden layer
      hidden_inputs = np2.dot(self.wih.T, inputs)
      # calculate the signals emerging from hidden layer
      hidden_outputs = self.activation_function(hidden_inputs)
      
      # calculate signals into final output layer
      final_inputs = np2.dot(self.who.T, hidden_outputs)
      # calculate the signals emerging from final output layer
      final_outputs = self.activation_function(final_inputs)
    
      """
      #to show the final output layer (the result)
      f_o = np2.array(final_outputs).reshape(28, 28)
      print(np2.shape(f_o))
      plt.imshow(f_o)
      plt.show(block= False)
      plt.pause(1)
      plt.close()
      """ 
      return final_outputs

      pass
