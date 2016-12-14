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
  double weight = 0;
  double delta = 0;
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
  double value = 0;
  double target = 0;
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

  bool wSet = false;
  bool iSet = false;
  bool fSet = false;
  bool tSet = false;

  private ulong[] m_size = [2,2,2];

  private const double   LEARN_RATE = 0.05;//0.3
  private const double   LEARN_ALPHA = 0.3;
  private const double[] WEIGHT_RANGE = [-0.05, 0.05];

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
      /* Link Weight to l/r neurons */
      a[$-1].lefts ~= &m_neurons[n][$-1];
      m_neurons[n][$-1].rights ~= &a[$-1];
      foreach(ref b; m_neurons[n+1][0..$-1]) {
        m_weights[n][$-1].rights ~= &b;
        b.lefts ~= &m_weights[n][$-1];
      }

      /* Set bias on left neuron */
      m_neurons[n][$-1].bias = true;
    }
    reset;
  }

  public auto calcError() {
    FPTemporary!double totalError = 0.0; // Init TE
    if(!fSet) return totalError;
    foreach(i, a; m_neurons[$-1])
      totalError += (1.0/2.0) * (a.target-a.value)^^2;
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

  public bool feedForward() {
    if(!wSet || !iSet || !tSet || fSet) return false;

    /* Sigmoid Lambda */
    auto sigmoid = (double pa) => 1/(1+E^^(-pa));

    /* Iterate through neuron layers, 1..n */
    foreach(n, ref a; m_neurons[1..$]) {
      foreach(ref b; a[0..$]) {
        if(b.bias) continue;
        foreach(c; b.lefts){
          enforce(c.lefts.length == 1);
          if(c.lefts[0].bias) b.value += c.weight;
          else b.value += c.weight * c.lefts[0].value;
        }
        b.value = sigmoid(b.value);
      }
    }
    fSet = true;
    return true;
  }

  public bool backProp() {
    if(!fSet) return false;

    foreach_reverse(n, ref a; m_weights){
      foreach(x, ref b; a) {
        if(n == m_weights.length-1) {
          writeln("JK", b.toStringF());
          writeln("o*(1-o)*(t-o)");
        } else {
          writeln("IJ",b);
          writeln("o*(1-o)*d*Sum(w)");
        }
      }
    }

    //FPTemporary!double[][] weightDelta;
    //weightDelta.length = m_weights.length;
    //foreach(i, a; m_weights)
    //  weightDelta[i].length = a.length;
    //FPTemporary!double[] biasDelta;
    //biasDelta.length = m_biases.length;
    //FPTemporary!double[] deltaNew;
    //FPTemporary!double[] deltaOld;
    //FPTemporary!double biasOld;

    /* Index Finders */
    //auto OKayI = (ulong pn, ulong px) => px/m_layers[pn].length;
    //auto OJayI = (ulong pn, ulong px) => px%m_layers[pn].length;

    /* Object Finders */
    //auto OKay = (ulong pn, ulong px)
    //  { return m_layers[pn+1][OKayI(pn,px)]; };
    //auto OJay = (ulong pn, ulong px)
    //  { return m_layers[pn][OJayI(pn,px)]; };
    //auto Target = (ulong pn, ulong px)
    //  { return m_targets[OKayI(pn,px)]; };
    //auto oldDelta = (ulong pn, ulong px)
    //  { return deltaOld[OKayI(pn, px)]; };

    /* Formulas */
    //auto deltak = (double ok, double tgt)
    //  { return ok*(1-ok)*(tgt-ok); };
    //auto deltaj = (double ok, double wSum, double old)
    //  { return ok*(1-ok)*wSum*old; };
    //auto DeltaK = (ulong pn, ulong px)
    //  { return deltak(OKay(pn,px), Target(pn,px)); };
    //auto DeltaJ = (ulong pn, ulong px)
    //  { return deltaj(OKay(pn,px), sum(m_weights[pn+1]), oldDelta(pn, px)); };

    //auto DeltaW = (ulong pn, ulong px, double delegate(ulong, ulong) delta)
    //  { return m_learnRate*delta(pn,px)*OJay(pn,px); };

    /* Avgs */
    //auto AvgDeltaK = (ulong pn) {
    //  FPTemporary!double[] deltas;
    //  foreach(i, o; m_layers[pn+1]) deltas ~= deltak(o, m_targets[i]);
    //  return sum(deltas)/deltas.length;
    //};
    //auto AvgDeltaJ = (ulong pn) {
    //  FPTemporary!double[] deltas;
    //  foreach(i, o; m_layers[pn+1])
    //    deltas ~= deltaj(o, sum(m_weights[pn+1]), biasOld);
    //  return sum(deltas)/deltas.length;
    //};

    /* Bias Function */
    //auto DeltaBias = (ulong pn, double delegate(ulong) avg)
    //  { return m_learnRate*avg(pn)*1.0; };

    /* Process the Layers */
    //foreach_reverse(n, a; m_weights){
    //  if(n == m_weights.length-1) {/* jk iteration */
    //    foreach(x, b; a) {
    //      weightDelta[n][x] = DeltaW(n,x, DeltaK);
    //      deltaOld ~= DeltaK(n,x);
    //    }
    //    /* Do the same for the biases */
    //    biasDelta[n] = DeltaBias(n, AvgDeltaK);
    //    biasOld = AvgDeltaK(n);
    //  } else { /* ij iterations */
    //    foreach(x, b; a) {
    //      weightDelta[n][x] = DeltaW(n,x, DeltaJ);
    //      deltaNew ~= DeltaJ(n,x);
    //    }
    //    /* Do the same for the biases */
    //    biasDelta[n] = DeltaBias(n, AvgDeltaJ);
    //    biasOld = AvgDeltaJ(n);
    //    /* End of cycle maintenance */
    //    deltaOld = deltaNew.dup;
    //    deltaNew.length = 0;
    //  }
    //}

    /* Apply deltas *//* Apply Momentum */
    //foreach(y, ref a; m_weights) foreach(x, ref b; a)
    //  b += weightDelta[y][x] + alpha * m_weightDelta[y][x];

    //foreach(y, ref a; m_biases)
    //  a += biasDelta[y] + alpha * m_biasDelta[y];

    //m_weightDelta = weightDelta.dup;
    //m_biasDelta = biasDelta.dup;
    //fSet = false;
    return false;
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