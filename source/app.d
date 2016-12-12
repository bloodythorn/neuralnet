import std.stdio;
import core.time;
import neuralnet;
import parser;

void main(string[] args)
{
  writeln(args);
  auto inputs = Digit.WIDTH * Digit.HEIGHT;
  auto outputs = 5UL;
  auto layers = [10UL];
  NeuralNet net = new NeuralNet(inputs, outputs, layers);
  net.randWeights;

  string fName = "TrainingData.dat";
  auto data = parseInput(fName);

  ulong index = 0;
  ulong target = 0;
  foreach(i, a; data) {
    if(a.m_value == target) {index = i; break;}
  }
  auto subject = data[index];
  writeln(subject);
  writeln(subject.test);

  if(!net.setInputs(subject.toInput)) writeln("Did not accept inputs");
  if(!net.setTargets(getTarget(subject.m_value, outputs)))
    writeln("Did not accept Targets");
  writeln("Initial Setup");
  net.feedForward;
  writeln(net);
  auto cycles = 10000L;
  for(int i = 0; i < cycles;++i) {
    auto result = net.train(cycles);
    writeln("After ", cycles, " cycles: ", result);
    writeln(net.m_outputs);
  }
}