#!/usr/bin/env python3
from net import *
import matplotlib.pyplot as plt
import time 
 

i_node = 784
h_nodes = 100
o_nodes = 10
l_rate = 0.1

# test the neural network
# scorecard for how well the network performs, initially empty
scorecard = []

#initialize the neural net
n = Net(i_node, h_nodes, o_nodes, l_rate)
  

def trainNetwork():
  train_file = open("/Users/KerimErekmen/Developer/Python/MachineLearing/handwritten/mnist_train_100.csv", "r")
  train_data = train_file.readlines()
  #print(train_data)
  train_file.close()

  epochs = 100
  for i in range(epochs):
      for run in train_data:
        vals = run.split(",")
        inputs = (np2.asfarray(vals[1:]) / 255.0 * 0.99) + 0.01
        targets = np2.zeros(o_nodes) + 0.01 
        targets[int(vals[0])] = 0.99
        n.train(inputs, targets)
      pass
  pass

def queryNetwork():
  global scorecard
  # load the mnist test data CSV file into a list
  test_data_file = open("/Users/KerimErekmen/Developer/Python/MachineLearing/handwritten/mnist_test_10.csv", "r")
  test_data_list = test_data_file.readlines()
  test_data_file.close()


  # go through all the records in the test data set
  for record in test_data_list:
      print(type(record))
      all_values = record.split(",")
      print(all_values)
      # correct answer is first value
      correct_label = int(all_values[0])
      # scale and shift the inputs
      inputs_list = (np2.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
      # query the network
      outputs = n.query(inputs_list)
      
      #the index of the highest value corresponds to the label
      label = np2.argmax(outputs)
      print(label, type(label))
      #append correct or incorrect to list
      if (label == correct_label):
          #network's answer matches correct answer, add 1 to scorecard
          scorecard.append(1)
      else:
          # networks answer doesn't match correct answer, add 0 to scorecard
          scorecard.append(0)
          pass
  pass 
  print(scorecard)

if __name__ == "__main__":
  trainNetwork()
  queryNetwork()


