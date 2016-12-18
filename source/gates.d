module gates;

import core.time;
import std.stdio;
import std.typecons: Tuple;

import neuralnet;

void andGate() {
  /* Let's try another. This time, let's do some circuits */
  /* We'll start with a AND */
  auto net = new NeuralNet([2,10,1]);

  /* We need some training samples. AND has four states. */
  double[][] andGate = [
    [0.1,0.1,0.1], //  false & false = false
    [0.1,0.9,0.1], //  false & true  = false
    [0.9,0.1,0.1], //  true  & false = false
    [0.9,0.9,0.9]]; // true  & true  = true
    net.randWeights;

  /* Train it */
  trainAnds(net, andGate, 1.0e-10);

  /* One more time so we can view the output */
  foreach(a; andGate) {
    writeln(a[0..$-1]);
    if(!net.setInputs(a[0..$-1])) { writeln("Inputs!"); return; }
    if(!net.setTargets([a[$-1]])) { writeln("Targets!"); return; }
    net.feedForward;
    writeln("  ", net.getOutputs);
    writeln("  ", net.calcError);
  }
  /* As you will see from the output, it will only read 0.9, when both inputs
  are 0.9. */
}

void trainAnds(NeuralNet p_net, double[][] p_gates, double p_prec) {
  bool done = false;
  Tuple!(double, long)[] results;
  auto start = MonoTime.currTime;
  auto last = MonoTime.currTime;

  while(!done) {
    foreach(a; p_gates) {
      if(!p_net.setInputs(a[0..$-1])) { writeln("Inputs!"); return; }
      if(!p_net.setTargets([a[$-1]])) { writeln("Targets!"); return; }
      if(!p_net.feedForward) { writeln("Feed"); return;}
      if(p_net.calcError > p_prec)
        results ~= p_net.train(p_prec);
    }
    auto current = MonoTime.currTime;
    auto diff = (current - last);
    auto act = seconds(5);
    if(diff > act) {
      writeln(
        "Training in Progress...",
        " Time:",  current - start);
      last = current;
    }
    //writeln("End of round: ", results.length, " trains");
    //if(results.length > 0) writeln("  Last result:", results[$-1]);
    if(results.length == 0) done = true;
    else results.length = 0;
  }
}

void fullAdder() {
  auto net = new NeuralNet([3, 4, 2]);
  //https://en.wikipedia.org/wiki/Adder_(electronics)
  // Columns are: Ain Bin Cin, Cout S
  double[][] fullAdder = [
    [0.1, 0.1, 0.1, 0.1, 0.1],
    [0.0, 0.0, 0.9, 0.1, 0.9],
    [0.1, 0.9, 0.1, 0.1, 0.9],
    [0.9, 0.1, 0.1, 0.1, 0.9],
    [0.1, 0.9, 0.9, 0.9, 0.1],
    [0.9, 0.1, 0.9, 0.9, 0.1],
    [0.9, 0.9, 0.1, 0.9, 0.1],
    [0.9, 0.9, 0.9, 0.9, 0.9]];
  net.randWeights;

  trainAdder(net, fullAdder, 1.0e-4);

  foreach(a; fullAdder) {
    writeln(a[0..$-2], a[$-2..$]);
    if(!net.setInputs(a[0..$-2])) { writeln("Inputs!"); return; }
    if(!net.setTargets(a[$-2..$])) { writeln("Targets!"); return; }
    net.feedForward;
    writeln("  ", net.getOutputs, " ", net.calcError);
  }
}

void trainAdder(NeuralNet p_net, double[][] p_gates, double p_prec) {
  bool done = false;
  Tuple!(double, long)[] results;
  auto start = MonoTime.currTime;
  auto last = MonoTime.currTime;

  while(!done) {
    foreach(a; p_gates) {
    if(!p_net.setInputs(a[0..$-2])) { writeln("Inputs!"); return; }
    if(!p_net.setTargets(a[$-2..$])) { writeln("Targets!"); return; }
      if(!p_net.feedForward) { writeln("Feed"); return;}
      if(p_net.calcError > p_prec)
        results ~= p_net.train(p_prec);
    }
    auto current = MonoTime.currTime;
    auto diff = (current - last);
    auto act = seconds(5);
    if(diff > act) {
      writeln(
        "Training in Progress...",
        " Time:",  current - start);
      last = current;
    }
    //writeln("End of round: ", results.length, " trains");
    //if(results.length > 0) writeln("  Last result:", results[$-1]);
    if(results.length == 0) done = true;
    else results.length = 0;
  }
}