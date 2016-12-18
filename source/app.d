import std.stdio;

import neuralnet;
import parser;

void main(string[] args)
{
  /* The name we'll store and retrieve our weights from */
  auto wFileName = "Weights.dat";

  /* Parse our training/recognition data */
  auto digits = parseInput("TrainingData.dat");

  /* Initiate our Neural Net */
  NeuralNet net = new NeuralNet([72,10,5]);

  /* Set a precision to work to/with */
  const double precision = 1.0e-4;

  /* Loads weights, or will generate new ones if it can't read old ones */
  writeln("Loading Weights");
  auto weights = loadWeights(wFileName);
  if(weights.length != net.getWeightsLength) {
    writeln("Could not load weights, generating new ones...");
    net.randWeights;
  } else net.setWeights(weights);

  /* Print beginning state of neural net */
  writeln(net);

  /* Train */
  train(net, digits, precision);

  /* Test each digit for accuracy, print results as hit/miss */
  auto accuracy = testResults(net, digits, precision);
  writefln("Hits/Misses: %s/%s ", accuracy[0], accuracy[1]);

  /* Finally, save weights so they can be used later */
  saveWeights(net.getWeights, wFileName);
}

