import core.time;
import std.algorithm;
import std.file;
import std.json;
import std.random;
import std.stdio;
import std.typecons: Tuple;
import neuralnet2;
import parser;

void main(string[] args)
{
  auto wFileName = "Weights.dat";
  auto digits = parseInput("TrainingData.dat");
  NeuralNet net = new NeuralNet([72,10,5]);
  const double precision = 1.0e-4;

  writeln("Loading Weights");
  auto weights = loadWeights(wFileName);
  if(weights.length != net.getWeightsLength) {
    writeln("Could not load weights, generating new ones...");
    net.randWeights;
  } else net.setWeights(weights);

  writeln(net);
  train2(net, digits, precision);
  writeln(testResults(digits, net, precision));

  saveWeights(net.getWeights, wFileName);
}

double[] loadWeights(string p_name) {
  double[] output;
  if(exists(p_name)) {
    File wFile;
    wFile.open(p_name, "r");
    if(wFile.isOpen) {
    string wStr;
    while(!wFile.eof()) wStr ~= wFile.readln();
    wFile.close();
    JSONValue wJSON = parseJSON(wStr);
    for(int i = 0; i < wJSON.array.length; ++i) output ~= wJSON[i].floating;
    }
  }

  return output;
}

void saveWeights(double[] p_wgts, string p_name) {
  JSONValue jj = JSONValue(p_wgts);
  auto file = new File("Weights.dat", "w");
  file.rawWrite(jj.toString);
  file.close;
}

Tuple!(uint, uint) testResults(
  in Digit[] p_in,
  NeuralNet p_net,
  double p_prec) {
  Tuple!(uint, uint) output;
  uint hit, miss;
  foreach(a; p_in) {
    if(!p_net.setInputs(a.getInput)) {
      writeln("Inputs not accepted"); break;
    }
    if(!p_net.setTargets(a.getTarget(5))) {
      writeln("Targets not accepted"); break;
    }
    p_net.feedForward;
    if(p_net.calcError > p_prec) output[1]++;
    else output[0]++;
  }
  return output;
}

void train(ref NeuralNet p_net, in Digit[] p_digs, in double p_prec) {
  const uint trains = 1;
  const uint checks = 2;
  auto start = MonoTime.currTime;
  auto last = MonoTime.currTime;
  Tuple!(double, double)[] results;

  while(true) {
    static uint count = 0;
    auto current = MonoTime.currTime;
    auto diff = (current - last);
    auto act = seconds(5);
    if(diff > act) {
      if(results.length > 0) {
        writefln(
          "Num %s b/a %e/%e",
          p_digs[count].m_value,
          results[$-1][0], results[$-1][1]);
        last = current;
      }
    }
    bool madeIt = true;
    static bool[] checkIt;
    if(!p_net.setInputs(p_digs[count].getInput)) {
      writeln("Inputs not accepted"); break;
    }
    if(!p_net.setTargets(p_digs[count].getTarget(5))) {
      writeln("Targets not accepted"); break;
    }
    p_net.feedForward;
    if(p_net.calcError > p_prec) {
      results ~= p_net.train(trains);
      madeIt = false;
    }
    if(++count >= p_digs.length) {
      if(madeIt) checkIt ~= true;
      if(checkIt.length >= checks) {
        if(reduce!((a,b)=> a && b)(true, checkIt)) break;
        checkIt.length = 0;
      }
      writefln("Round end, %s trains.", results.length);
      results.length = 0;
      madeIt = true;
      count = 0;
    }
  }
}

void train2(ref NeuralNet p_net, in Digit[] p_digs, in double p_prec) {
  if(p_digs.length == 0) return; //Nothing to do
  bool done = false;
  auto start = MonoTime.currTime;
  auto last = MonoTime.currTime;
  immutable uint checks = 2;
  uint round = 0;
  while(!done)
  {
    Tuple!(double, long)[] results;

    foreach(a; p_digs) {
      if(!p_net.setInputs(a.getInput)) {
        writeln("Inputs not accepted"); break;
      }
      if(!p_net.setTargets(a.getTarget(5))) {
        writeln("Targets not accepted"); break;
      }
      p_net.feedForward;
      if(p_net.calcError > p_prec) {
        write("Training a ", a.m_value);
        results ~= p_net.train(p_prec);
        writeln(" - ", results[$-1]);
      }
    }

    writefln("Round %s end, %s trains.", round++, results.length);
    static uint checksPassed;
    if(results.length == 0) checksPassed++;
    else checksPassed = 0;
    if(checksPassed >= checks) done = true;
  }
}