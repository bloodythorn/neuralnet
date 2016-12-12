module neuralnet2;

import std.algorithm;
import std.conv;
import std.exception: enforce;
import std.math;
import std.numeric;
import std.random;
import std.typecons;
import core.time;

import std.stdio;

struct Weight {
  double weight;
  Neuron*[] lefts;
  Neuron*[] rights;
  string toString() { return to!string(weight); }
  string toStringF() {
    return to!string(lefts.length) ~
           "<(" ~ to!string(weight) ~ ")>" ~
           to!string(rights.length);
  }
}

struct Neuron {
  bool bias = false;
  private double value;
  Weight*[] lefts;
  Weight*[] rights;
  string toString() { return ((bias) ? "bias" : to!string(value)); }
  string toStringF() {
    return to!string(lefts.length) ~
           "<(" ~ ((bias) ? "bias" : to!string(value)) ~
           ")>" ~
           to!string(rights.length);
  }
}

class NeuralNet {

  Neuron[][] m_neurons;
  Weight[][] m_weights;
  double[]   m_targets;

  bool wSet = false;
  bool iSet = false;
  bool fSet = false;
  bool tSet = false;

  private const double[] WEIGHT_RANGE = [-0.1, 0.1];

  private ulong[] m_size = [2,2,2];

  public this() { init; }

  public this(ulong[] p_size) {
    enforce(p_size.length > 2);
    m_size = p_size.dup; init;
  }

  private void init() {

    auto getXNeu = (long pn, long pi)
      { return &m_neurons[pn][pi%(m_neurons[pn].length-1)]; };
    auto getYNeu = (long pn, long pi)
      { return &m_neurons[pn+1][pi/(m_neurons[pn].length-1)]; };

    /* size */
    m_neurons.length = m_size.length;
    foreach(i, ref a; m_neurons)
      a.length = (i != m_neurons.length-1) ? m_size[i]+1 : m_size[i];
    m_weights.length = m_size.length-1;
    foreach(i, ref a; m_weights)
      a.length = m_size[i] * m_size[i+1] +1;

    /* link up normal neurons */
    foreach(n, ref a; m_weights)
      foreach(i, ref b; a)
        if(i != a.length-1) {
          b.lefts ~= getXNeu(n,i);
          getXNeu(n,i).rights ~= &b;
          b.rights ~= getYNeu(n,i);
          getYNeu(n,i).lefts ~= &b;
        }

    /* Link up Biases */
    foreach(n, ref a; m_weights) {
      a[$-1].lefts ~= &m_neurons[n][$-1];
      m_neurons[n][$-1].bias = true;
      m_neurons[n][$-1].rights ~= &a[$-1];
      foreach(ref b; m_neurons[n+1][0..$-1])
        m_weights[n][$-1].rights ~= &b;
    }
    m_weights[$-1][$-1].rights ~= &m_neurons[$-1][$-1];
  }

  public auto calcError() {
    FPTemporary!double totalError = 0.0; // Init TE
    if(!fSet) return totalError;
    foreach(i, a; m_neurons[$-1])
      totalError += (1.0/2.0) * (m_targets[i]-a.value)^^2;
    return totalError;
  }

  public void randWeights() {
    Random gen;
    foreach(ref a; m_weights)
      foreach(ref b; a)
        b.weight = uniform(WEIGHT_RANGE[0], WEIGHT_RANGE[1], gen);
    wSet = true;
  }

  override string toString() {
    const ulong LIMIT = 10;
    string output = "-----------------------------------\n";
    output ~= "Inputs/Outputs/Hiddens/Outputs: ";
    output ~= to!string(m_size[0]) ~ "/"
            ~ to!string(m_size[1..$-1]) ~ "/"
            ~ to!string(m_size[$-1]) ~ " ";
    ulong[] leng;
    foreach(a; m_neurons)
      leng ~= a.length;
    output ~= "Actual Layer Lengths: " ~ to!string(leng) ~ "\n";

    output ~= "Input Values : (Limited to " ~ to!string(LIMIT) ~ ")\n";
    output ~= "  [ ";
    foreach(x, a; m_neurons[0]) {
      if(x > LIMIT) { output ~= "..."; break;}
      output ~= to!string(a) ~ " ";
    }
    output ~= "]" ~ "\n";

    output ~= "Hidden Values : (Limited to " ~ to!string(LIMIT) ~ ")\n";
    foreach(x, a; m_neurons) {
      if(x > LIMIT) { output ~= "..."; break;}
      output ~= "  [ ";
      foreach(y, b; a) {
        if(y > LIMIT) { output ~= "..."; break;}
        output ~= to!string(b) ~ " ";
      }
      output ~= "]" ~ "\n";
    }

    output ~= "Output Values :\n";
    output ~= "  [ ";
    foreach(x, a; m_neurons[$-1])
      output ~= to!string(a) ~ " ";
    output ~= "]" ~ "\n";

    leng.length = 0;
    foreach(a; m_weights)
      leng ~= a.length;
    output ~= "Weight Layers: " ~ to!string(leng) ~ "\n";
    output ~= "Weight Values: (Limited to " ~ to!string(LIMIT) ~ ")\n";
    foreach(x, a; m_weights) {
      if(x > LIMIT) { output ~= "..."; break;}
      output ~= "  [ ";
      foreach(y, b; a) {
        if(y > LIMIT) { output ~= "..."; break;}
        output ~= to!string(b) ~ " ";
      }
      output ~= "]" ~ "\n";
    }

    output ~= "Target Values: " ~ to!string(m_targets) ~ "\n";
    output ~= "Current Error: " ~ to!string(calcError) ~ "\n";
    output ~= "Weights Set  : " ~ to!string(wSet) ~ "\n";
    output ~= "Inputs  Set  : " ~ to!string(iSet) ~ "\n";
    output ~= "FedForward   : " ~ to!string(fSet) ~ "\n";
    output ~= "Target Set   : " ~ to!string(tSet) ~ "\n";
    output ~= "-----------------------------------";
   return output;
  }

}