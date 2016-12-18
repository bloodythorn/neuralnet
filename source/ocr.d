module ocr;

/**
  @brief Parser / Helper Module

  Description: This module contains functions that custom tailors the data
    for the example used in main.

  */

import core.time;
import std.conv;
import std.file;
import std.json;
import std.stdio;
import std.string;
import std.typecons: Tuple;

import neuralnet;

void ocrRun() {
  /* The name we'll store and retrieve our weights from */
  auto wFileName = "Weights.dat";

  /* Parse our training/recognition data */
  auto digits = parseInput("TrainingData.dat");

  /* Initiate our Neural Net */
  NeuralNet net = new NeuralNet([72,10,5]);

  /* Set a precision to work to/with */
  const double precision = 9.0e-5;

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

/**
  @brief  Digit Structure

   Description: This structure is a model for the ascii digits

  */
struct Digit {
  static const ulong WIDTH = 8;
  static const ulong HEIGHT = 9;
  bool[WIDTH][HEIGHT] m_rep;
  ulong m_value = 0;
  string toString() {
      string output = to!string(m_value) ~ "|";
      foreach(a; m_rep[0])
        if(a) output ~= "1";
        else output ~= "0";
      output ~= "|\n";
      foreach(a; m_rep[1..$]) {
        output ~= " |";
        foreach(b; a)
          if(b) output ~= "1";
          else output ~= "0";
        output ~= "|\n";
      }
      return output;
  }
  bool test() {
    bool output = false;
    foreach(a; m_rep) foreach(b; a) output |= b;
    return output;
  }
  const double[] getInput() {
    double[] output;
    foreach(a; m_rep) foreach(b; a)
      if(b) output ~= 0.99;
      else output ~= 0.01;
    return output;
  }
  const double[] getTarget(uint p_size) {
    double[] output;
    for(uint i = 0; i < p_size;++i)
      if(i == m_value) output ~= 0.99;
      else output ~= 0.01;
    return output;
  }
}

/**
  @brief Parses Input

  Description: This function, given a file name will parse the given digits
    into a usable data structure.

  */
Digit[] parseInput(string p_file) {
  Digit[] output;

  File file = File(p_file);
  if(!file.isOpen) return output;

  while(!file.eof){
    Digit temp = parseDigit(file);
    if(temp.test)
      output ~= temp;
    else return output;
  }

  return output;
}

/**
  @brief Parse Single Digit

  Description: This is the helper function to parseInput. It parses each digit
    one at a time.
  */
Digit parseDigit(File p_file) {

  /* Setup */
  Digit output;
  string line;
  uint row = 0;
  uint col= 0;
  /* Iterate through each line */
  while (!p_file.eof) {
    line = strip(p_file.readln);
    if (row >= 0 && row < Digit.HEIGHT) { /* Normal Row */
      if (line.length == Digit.WIDTH) { /* Normal Row Size */
        while (line.length != 0) {
          switch (line[0]) {
            case '0': {
              output.m_rep[row][col++] = false;
              break;
            }
            case '1': {
              output.m_rep[row][col++] = true;
              break;
            }
            default: {
              output = Digit();
              return output;
            }
          }
          line = line[1..$];
        }
        col = 0;
        row++;
      } else {
        output = Digit();
        return output;
      }
    }
    else if (row == Digit.HEIGHT) { /* Read Number at end */
      if (line.length == 1) {
        ulong value = parse!ulong(line);
        output.m_value = value;
        if(line.length > 0) {
          output = Digit();
          return output;
        }
      }
      else {
        output = Digit();
        return output;
      }
      row++;
    }
    else {
      /* This can be only whitespace */
      if (line.length != 0) {
        output = Digit();
        return output;
      }
      else break;
    }
  }
  if (row != 10 || col != 0) {
    output = Digit();
    return output;
  }
  return output;
}

/**
  @brief Save/Load Weights

  Description: Given a filename and/or flat array, these functions will save
    or load it to the given file name so the weights of the net can be stored
    and retrieved.
  */
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
      if(wJSON.type != JSON_TYPE.ARRAY) return output;
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

/**
  @brief Trainer Function

  Description: Given a net, some digits to train and a precision to train to,
    this function will train the neurtal net until no digit exceeds the given
    threshold.
  */
void train(ref NeuralNet p_net, in Digit[] p_digs, in double p_prec) {
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
        writefln("Training a %s ", a.m_value);
        results ~= p_net.train(p_prec);
        writefln(
          "  - from %e to %e in %s epochs ",
          results[$-1][0], p_prec, results[$-1][1]);
      }
    }

    writefln("Round %s end, %s trains.", round++, results.length);
    static uint checksPassed;
    if(results.length == 0) checksPassed++;
    else checksPassed = 0;
    if(checksPassed >= checks) done = true;
  }
}

/**
  @brief Test Digits

  Description: Given a net, a list of digits, and a precision, this function
    will run through each and tally hits or misses, dictated by the threshold
    given.
  */
Tuple!(uint, uint) testResults(
  NeuralNet p_net,
  in Digit[] p_in,
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

