module neuralnet;

import std.algorithm;
import std.conv;
import std.math;
import std.numeric;
import std.random;
import std.typecons;
import core.time;

import std.stdio;

class NeuralNet {

  private double[][] m_layers;
  private double[][] m_weights;
  private double[][] m_weightDelta;

  private double[]   m_targets;
  private double[]   m_biases;
  private double[]   m_biasDelta;

  private bool wSet = false;
  private bool iSet = false;
  private bool fSet = false;
  private bool tSet = false;

  private uint   m_nInputs   = 2;
  private uint   m_nOutputs  = 2;
  private uint[] m_nLayers   = [2];
  private double  m_learnRate = 0.05;
  private double  alpha       = 0.01;

  private const double[] WEIGHT_RANGE = [-0.1, 0.1];

  public @property auto m_inputs()
    { assert(m_layers.length > 0); return m_layers[0]; }
  private @property auto m_inputs(double[] p_in)
    { return m_layers[0] = p_in.dup ;}
  public @property auto m_outputs()
    { assert(m_layers.length > 1); return m_layers[$-1]; }
  private @property auto m_outputs(double[] p_in)
    { assert(m_layers.length > 1); return m_layers[$-1] = p_in; }
  public @property auto m_hiddens()
    { assert(m_layers.length > 2); return m_layers[1..$-1]; }
  private @property auto m_hiddens(double[][] p_in)
    { assert(m_layers.length > 2); return m_layers[1..$-1] = p_in; }

  public this() { init; }

  public this(uint p_in, uint p_out, uint[] p_nLayers) {
    setSize(p_in, p_out, p_nLayers);
    init;
  }

  public void randWeights() {
    Random gen;
    foreach(ref a; m_weights)
      foreach(ref b; a)
        b = uniform(WEIGHT_RANGE[0], WEIGHT_RANGE[1], gen);
    m_biases.length = m_weights.length;
    foreach(ref a; m_biases)
      a = uniform(WEIGHT_RANGE[0], WEIGHT_RANGE[1], gen);
    initDeltas;
    wSet = true;
  }

  private void initDeltas(){
    foreach(ref a; m_weightDelta)
      foreach(ref b; a) b = 0.0;
    foreach(ref a; m_biasDelta) a = 0.0;
  }

  private void init() {
    m_layers.length = 0;
    m_layers.length++;
    m_layers[$-1].length = m_nInputs;
    foreach(a; m_nLayers) {
      m_layers.length++;
      m_layers[$-1].length = a;
    }
    m_layers.length++;
    m_layers[$-1].length = m_nOutputs;

    m_weights.length = 0;
    m_weights.length = 1 + m_nLayers.length;
    m_weightDelta.length = 1 + m_nLayers.length;
    foreach(i, ref a; m_weights) {
      a.length = m_layers[i].length * m_layers[i+1].length;
      m_weightDelta[i].length = m_layers[i].length * m_layers[i+1].length;
    }

    m_biases.length = 0;
    m_biases.length = m_weights.length;
    m_biasDelta.length = 0;
    m_biasDelta.length = m_weights.length;
    wSet = iSet = fSet = tSet = false;
  }

  private void reset(){
    fSet = false;
    foreach(ref a; m_layers[1..$]) foreach(ref b; a) b = double.nan;
    initDeltas;
  }

  public void setSize(uint p_in, uint p_out, uint[] p_nLayers) {
    m_nInputs = p_in;
    m_nOutputs = p_out;
    m_nLayers = p_nLayers.dup;
    init;
  }

  public bool setInputs(in double[] p_in) {
    if(m_inputs.length != p_in.length) return false;
    reset;
    m_inputs = p_in.dup;
    iSet = true;
    return true;
  }

  public bool setWeights(double[][] p_weights,double[] p_bWeights) {
    /* Verify Input */
    if(p_weights.length != m_weights.length) return false;
    if(p_bWeights.length != m_weights.length) return false;
    foreach(i, a; m_weights)
      if(a.length != p_weights[i].length) return false;
    reset;
    m_weights = p_weights.dup;
    m_biases = p_bWeights.dup;
    initDeltas;
    wSet = true;
    return true;
  }

  public bool setTargets(double[] p_in) {
    if(p_in.length != m_nOutputs) return false;
    reset;
    m_targets = p_in.dup;
    tSet = true;
    return true;
  }

  public bool feedForward() {
    if(!wSet || !iSet || !tSet || fSet) return false;

    /* Sigmoid Lambda */
    auto sigmoid = (double pa) => 1/(1+E^^(-pa));
    auto wi = (uint pz, uint px, uint py) => py*m_layers[pz].length+px;

    /* Cycle Through each layer n+1 */
    foreach(n, ref a; m_layers[1..$])
      foreach(y, ref b; a) { /* y/y b node with each c node */
        b = 0.0;
        foreach(x, c; m_layers[n])
          b += c*m_weights[n][wi(n, x, y)];
        b+= m_biases[n];
        b = sigmoid(b);
      }

    fSet = true;
    return true;
  }

  public auto calcError() {
    FPTemporary!double totalError = 0.0; // Init TE
    if(!fSet) return totalError;
    foreach(i, a; m_outputs) totalError += (1.0/2.0) * (m_targets[i]-a)^^2;
    return totalError;
  }

  public bool backProp() {
    if(!fSet) return false;
    return true;
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
    //auto OKayI = (uint pn, uint px) => px/m_layers[pn].length;
    //auto OJayI = (uint pn, uint px) => px%m_layers[pn].length;

//    /* Object Finders */
//    auto OKay = (uint pn, uint px)
//      { return m_layers[pn+1][OKayI(pn,px)]; };
//    auto OJay = (uint pn, uint px)
//      { return m_layers[pn][OJayI(pn,px)]; };
//    auto Target = (uint pn, uint px)
//      { return m_targets[OKayI(pn,px)]; };
//    auto oldDelta = (uint pn, uint px)
//      { return deltaOld[OKayI(pn, px)]; };

//    /* Formulas */
//    auto deltak = (double ok, double tgt)
//      { return ok*(1-ok)*(tgt-ok); };
//    auto deltaj = (double ok, double wSum, double old)
//      { return ok*(1-ok)*wSum*old; };
//    auto DeltaK = (uint pn, uint px)
//      { return deltak(OKay(pn,px), Target(pn,px)); };
//    auto DeltaJ = (uint pn, uint px)
//      { return deltaj(OKay(pn,px), sum(m_weights[pn+1]), oldDelta(pn, px)); };

//    auto DeltaW = (uint pn, uint px, double function(uint, uint) delta)
//      { return m_learnRate*delta(pn,px)*OJay(pn,px); };

//    /* Avgs */
//    auto AvgDeltaK = (uint pn) {
//      FPTemporary!double[] deltas;
//      foreach(i, o; m_layers[pn+1]) deltas ~= deltak(o, m_targets[i]);
//      return sum(deltas)/deltas.length;
//    };
//    auto AvgDeltaJ = (uint pn) {
//      FPTemporary!double[] deltas;
//      foreach(i, o; m_layers[pn+1])
//        deltas ~= deltaj(o, sum(m_weights[pn+1]), biasOld);
//      return sum(deltas)/deltas.length;
//    };

//    /* Bias Function */
//    auto DeltaBias = (uint pn, double function(uint) avg)
//      { return m_learnRate*avg(pn)*1.0; };

//    /* Process the Layers */
//    foreach_reverse(n, a; m_weights){
//      if(n == m_weights.length-1) {/* jk iteration */
//        foreach(x, b; a) {
//          weightDelta[n][x] = DeltaW(n,x, DeltaK);
//          deltaOld ~= DeltaK(n,x);
//        }
//        /* Do the same for the biases */
//        biasDelta[n] = DeltaBias(n, AvgDeltaK);
//        biasOld = AvgDeltaK(n);
//      } else { /* ij iterations */
//        foreach(x, b; a) {
//          weightDelta[n][x] = DeltaW(n,x, DeltaJ);
//          deltaNew ~= DeltaJ(n,x);
//        }
//        /* Do the same for the biases */
//        biasDelta[n] = DeltaBias(n, AvgDeltaJ);
//        biasOld = AvgDeltaJ(n);
//        /* End of cycle maintenance */
//        deltaOld = deltaNew.dup;
//        deltaNew.length = 0;
//      }
//    }

//    /* Apply deltas *//* Apply Momentum */
//    foreach(y, ref a; m_weights) foreach(x, ref b; a)
//      b += weightDelta[y][x] + alpha * m_weightDelta[y][x];

//    foreach(y, ref a; m_biases)
//      a += biasDelta[y] + alpha * m_biasDelta[y];

 //   m_weightDelta = weightDelta.dup;
 //   m_biasDelta = biasDelta.dup;
 //   return true;
 //   fSet = false;
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
    output ~= ("Inputs: " ~ to!string(m_inputs.length) ~ "\n");
    output ~= "Outputs: " ~ to!string(m_outputs.length) ~ "\n";
    output ~= "Hidden Layers: " ~ to!string(m_hiddens.length) ~ "\n";
    output ~= "Hidden Layer Lengths: ";
    foreach(a; m_hiddens)
      output ~= to!string(a.length) ~ " ";
    output ~= "\n";
    output ~= "Ttl Layers: " ~ to!string(m_layers.length) ~ "\n";
    output ~= "Weight Layers: " ~ to!string(m_weights.length) ~ "\n";
    output ~= "Weight Lengths: ";
    foreach(a; m_weights)
      output ~= to!string(a.length) ~ " ";
    output ~= "\n";
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
    output ~= "Bias Values  : " ~ to!string(m_biases) ~ "\n";
    output ~= "Input Values :\n";
    output ~= "  [ ";
    foreach(x, a; m_inputs)
        output ~= to!string(a) ~ " ";
    output ~= "]" ~ "\n";
    output ~= "Hidden Values: " ~ to!string(m_hiddens) ~ "\n";
    output ~= "Output Values: " ~ to!string(m_outputs) ~ "\n";
    output ~= "Target Values: " ~ to!string(m_targets) ~ "\n";
    output ~= "Weight Deltas: (Limited to " ~ to!string(LIMIT) ~ ")\n";
    foreach(x, a; m_weightDelta) {
      if(x > LIMIT) { output ~= "..."; break;}
      output ~= "  [ ";
      foreach(y, b; a) {
        if(y > LIMIT) { output ~= "..."; break;}
        output ~= to!string(b) ~ " ";
      }
      output ~= "]" ~ "\n";
    }
    output ~= "Bias Deltas  : " ~ to!string(m_biasDelta) ~ "\n";
    output ~= "Current Error: " ~ to!string(calcError) ~ "\n";
    output ~= "Weights Set  : " ~ to!string(wSet) ~ "\n";
    output ~= "Inputs  Set  : " ~ to!string(iSet) ~ "\n";
    output ~= "Target Set   : " ~ to!string(tSet) ~ "\n";
    output ~= "FedForward   : " ~ to!string(fSet) ~ "\n";
    output ~= "-----------------------------------";
   return output;
  }
}
