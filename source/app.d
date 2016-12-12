import std.stdio;
import core.time;
import neuralnet2;
import parser;

void main(string[] args)
{
  auto size = [ Digit.WIDTH * Digit.HEIGHT, 5UL, 10UL];
  NeuralNet net = new NeuralNet([16,22,35,10]);
  writeln(net);
  net.randWeights;
  writeln(net);
}