import std.stdio;
import core.time;
import neuralnet;

void main()
{
  NeuralNet net = new NeuralNet();
  /* These weights will only work in a 2x2x2 config */
  //if(!net.setWeights(
  //  [ [0.15, 0.2, 0.25, 0.3], [0.4, 0.45, 0.5, 0.55]],
  //  [0.35, 0.6])) {
  //  writeln("Weights not accepted");
  //  writeln(net);
  //  return;
  //}

  net.randWeights;
  if(!net.setInput([0.05,0.10])) {
    writeln("Input not accepted");
    writeln(net);
    return;
  }
  if(!net.setTargets([0.01, 0.99])) {
     writeln("Targets not accepted");
     writeln(net);
     return;
  }

  writeln("Before:\n",net);
  auto threshold = 0.000000001;
  MonoTime start = MonoTime.currTime;
  auto result = net.train(threshold);
  MonoTime stop = MonoTime.currTime;
  writeln("Threshold: ", threshold);
  writeln("Cycles   : ", result);
  writeln("Error    : ", net.calcError);
  writeln("Completed in ", (stop - start));
  writeln("After:\n", net);
}

