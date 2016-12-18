module neuralnet2;

import std.algorithm;
import std.conv;
import std.exception: enforce;
import std.format;
import std.math;
import std.numeric;
import std.random;
import std.typecons;
import core.time;

import std.stdio;

class NeuralNet {

  private struct Weight {
    double weight = 0;
    double deltaW = 0;
    double deltaWOld = 0;
    string weightNum = "-";
    Neuron*[] lefts;
    Neuron*[] rights;

    void fire() {
      enforce(lefts.length == 1 && rights.length == 1);
      auto Ok = rights[0]; auto Oj = lefts[0];
      deltaW = LEARN_RATE*Ok.getDelta*Oj.value;
    }
    void mergeDelta() {
      weight += deltaW + MOMENTUM_ALPHA * deltaWOld;
      deltaWOld = deltaW;
      deltaW = 0;
    }

    string toString() {
      return weightNum ~ ":" ~ format(fmt,weight);
    }
    string toStringF() {
      return to!string(lefts.length) ~
             "<(" ~ weightNum ~ ":" ~
             to!string(weight) ~ ")>" ~
             to!string(rights.length);
    }
  }

  private struct Neuron {
    bool bias = false;
    double value = 0;
    double target = 0;
    double delta;
    bool dSet;
    string neuronNum = "+";
    Weight*[] lefts;
    Weight*[] rights;

    void fire() {
      auto ll = lefts.length; auto rl = rights.length;
      if((ll != 0 && rl != 0) || (ll != 0 && rl == 0))
        value = 1/(1+E^^(-value));
      if((ll == 0 && rl != 0) || (ll != 0 && rl != 0))
        foreach(x, ref a; rights) foreach(y, ref b; a.rights)
          b.value += (bias ? 1 : value) * a.weight;
      dSet = false; delta = 0;
    }

    double error() {
      if(target == double.nan) return 0;
      return (1.0/2.0) * (target-value)^^2;
    }

    double getDelta() {
      if(bias) return 0.0;if(dSet) return delta;
      delta = value*(1-value);
      if(rights.length == 0)
        delta = delta*(target-value);
      else
        delta =
          delta*reduce!((a,b)=>a+b.weight*b.rights[0].getDelta)(0.0, rights);
      dSet = true;
      return delta;
    }

    string toString() {
      return
        neuronNum ~ ":" ~ ((bias) ? "bias" : format(fmt,value));
    }
    string toStringF() {
      return to!string(lefts.length) ~
             "<(" ~ neuronNum ~ ":" ~
             ((bias) ? "bias" : to!string(value)) ~
             ")>" ~
             to!string(rights.length);
    }
  }

  private Neuron[][] m_neurons;
  private Weight[][] m_weights;

  bool wSet = false;
  bool iSet = false;
  bool fSet = false;
  bool tSet = false;

  /* TODO a way to externally control these. */
  private uint[] m_size = [2,2,2];
  private static auto fmt = "%+.2g";
  private static const double   MOMENTUM_ALPHA = 0.1;
  private static const double   LEARN_RATE = 0.3;
  private static const double[] WEIGHT_RANGE = [-1.0, 1.00];

  public this() { init; }

  public this(uint[] p_size) {
    enforce(p_size.length > 2);
    m_size = p_size.dup; init;
  }

  private void init() {

    auto getXNeu = (uint pn, uint pi)
      { return &m_neurons[pn][pi%(m_neurons[pn].length-1)]; };
    auto getYNeu = (uint pn, uint pi)
      { return &m_neurons[pn+1][pi/(m_neurons[pn].length-1)]; };

    /* size */
    m_neurons.length = 0;
    m_neurons.length = m_size.length;
    foreach(i, ref a; m_neurons)
      a.length = (i != m_neurons.length-1) ? m_size[i]+1 : m_size[i];
    m_weights.length = 0;
    m_weights.length = m_size.length-1;
    foreach(i, ref a; m_weights)
      a.length = m_size[i] * m_size[i+1] +1;

    /* link up normal neurons */
    foreach(n, ref a; m_weights)
      foreach(i, ref b; a[0..$-1]) {
        auto lXN = getXNeu(n,i);
        auto rYN = getYNeu(n,i);
          b.lefts ~= lXN;
          b.rights ~= rYN;
          lXN.rights ~= &b;
          rYN.lefts ~= &b;
        }

    /* Link up Biases */
    foreach(n, ref a; m_weights) {
      auto wgt = &a[$-1];
      auto lNeu = &m_neurons[n][$-1];
      wgt.lefts ~= lNeu;
      lNeu.rights ~= wgt;
      foreach(ref b; m_neurons[n+1][0..$-1]) {
        wgt.rights ~= &b;
        b.lefts ~= wgt;
      }
    }
    m_weights[$-1][$-1].rights ~= &m_neurons[$-1][$-1];
    m_neurons[$-1][$-1].lefts ~= &m_weights[$-1][$-1];

    /* Set bias on left neuron */
    foreach(a; m_neurons[0..$-1]) a[$-1].bias = true;

    /* Set Neuron/Weight numbers */
    uint count = 0;
    foreach(x, ref a; m_neurons) foreach(y, ref b; a) {
      if(b.bias) b.neuronNum = "b" ~ to!string(x);
      else b.neuronNum = "n" ~ to!string(count++);
    }
    count = 0;
    foreach(x, ref a; m_weights) foreach(y, ref b; a)
      b.weightNum = "w" ~ to!string(count++);

    reset;
  }

  public auto calcError() {
    FPTemporary!double totalError = 0.0; // Init TE
    if(!fSet) return totalError;
    foreach(a; m_neurons[$-1]) totalError += a.error;
    return totalError;
  }

  public void randWeights() {
    Random gen;
    foreach(ref a; m_weights)
      foreach(ref b; a)
        b.weight = uniform(WEIGHT_RANGE[0], WEIGHT_RANGE[1], gen);
    wSet = true;
  }

  private void reset(){
    fSet = false;
    foreach(ref a; m_neurons[1..$])
      foreach(ref b; a)
        b.value = 0;
  }

  public bool setInputs(in double[] p_in) {
    if(p_in.length != m_size[0]) return false;
    reset;
    foreach(i, ref a; m_neurons[0][0..$-1])
      a.value = p_in[i];
    iSet = true;
    return true;
  }

  public bool setTargets(double[] p_in) {
    if(p_in.length != m_size[$-1]) return false;
    reset;
    foreach(i, ref a; m_neurons[$-1])
      a.target = p_in[i];
    tSet = true;
    return true;
  }

  public bool setWeights(double[] p_in) {
    uint total = 0;
    foreach(n,a; m_size[0..$-1]) total += m_size[n] * m_size[n+1] +1;
    if(p_in.length != total) return false;
    total = 0;
    foreach(y, ref a; m_weights) foreach(x, ref b; a) b.weight = p_in[total++];
    wSet = true;
    return true;
  }

  public double[] getWeights() {
    double[] output;
    foreach(a; m_weights) {
      foreach(b; a) output ~= b.weight;
    }
    return output;
  }

  public uint getWeightsLength() {
    uint total = 0;
    foreach(n,a; m_size[0..$-1]) total += m_size[n] * m_size[n+1] +1;
    return total;
  }

  public void clearWeights() {
    foreach(ref a; m_weights) foreach(ref b; a) b.weight = 0.0;
  }

  public bool feedForward() {
    if(!wSet || !iSet || !tSet || fSet) return false;
    foreach(ref a; m_neurons) foreach(ref b; a) b.fire();
    fSet = true;
    return true;
  }

  public bool backProp() {
    if(!fSet) return false;

    /* Generate Weight Deltas */
    foreach_reverse(n, ref a; m_weights)
      foreach(y, ref b; a[0..$-1])
        b.fire();

    /* Apply Weight Deltas to Weights */
    foreach(ref a; m_weights) foreach(ref b; a) b.mergeDelta;

    reset; fSet = false;
    return true;
  }

  public auto trainCycle() {
    Tuple!(double, double) output;
    if(!wSet || !iSet || !tSet) return output;
    bool result;
    if(!fSet) {
      result = feedForward;
      if(!result) return output;
    }
    output[0] = calcError;

    result = backProp && feedForward;
    if(!result) return output;
    output[1] = calcError;
    return output;
  }

  public auto train(in ulong p_cyc) {
    Tuple!(double, double) output;
    bool result;
    if(!fSet) {
      result = feedForward;
      if(!result) return output;
    }
    output[0] = calcError;
    for(int i = 0; i < p_cyc; ++i)
      trainCycle;
    output[1] = calcError;
    return output;
  }

  public auto train(double p_thresh) {
    Tuple!(double, long) output;
    bool result;
    if(!fSet) {
      result = feedForward;
      if(!result) return output;
    }
    output[0] = calcError;
    auto cycle = 0;
    auto start = MonoTime.currTime;
    auto last = MonoTime.currTime;
    while(calcError > p_thresh){
      auto period = trainCycle;
      ++cycle;
      auto current = MonoTime.currTime;
      auto diff = (current - last);
      auto act = seconds(5);
      if(diff > act) {
        writeln(
          "Training in Progress...",
          " Start:", output[0],
          " Current:", calcError,
          " Target:", p_thresh,
          " Cycle:", cycle,
          " Time:",  current - start);
        last = current;
      }
    }
    output[1] = cycle;
    return output;
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
    foreach(x, a; m_neurons[1..$-1]) {
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

    output ~= "Target Values: [ ";
    foreach(a; m_neurons[$-1])
      output ~= to!string(a.target) ~ " ";
    output ~= "]" ~ "\n";

    output ~= "Current Error: " ~ to!string(calcError) ~ "\n";
    output ~= "Weights Set  : " ~ to!string(wSet) ~ "\n";
    output ~= "Inputs  Set  : " ~ to!string(iSet) ~ "\n";
    output ~= "FedForward   : " ~ to!string(fSet) ~ "\n";
    output ~= "Target Set   : " ~ to!string(tSet) ~ "\n";
    output ~= "-----------------------------------";
   return output;
  }
}