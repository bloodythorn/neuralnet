import std.stdio;
import core.time;
import neuralnet2;
import parser;

void main(string[] args)
{
  NeuralNet net = new NeuralNet([3,3,3,3]);
  net.randWeights;
  if(!net.setInputs([0.2,0.5, 0.1])) writeln("Inputs not accepted");
  if(!net.setTargets([0.9,0.1, 0.1])) writeln("Inputs not accepted");
  if(!net.feedForward) writeln("Feedforward Failed");
  if(!net.backProp) writeln("Backprop failed");
  writeln(net);
}