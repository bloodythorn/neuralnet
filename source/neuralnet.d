module neuralnet;

import std.conv;
import std.math;
import std.numeric;
import std.random;
import std.typecons;

class NeuralNet {
private:

  double[][] m_layers;
  double[][] m_weights;
  double[][] m_weightDelta;
  double[]   m_targets;
  double[]   m_biases;
  double[]   m_biasDelta;

  bool wSet = false;
  bool iSet = false;
  bool fSet = false;

  uint   m_nInputs = 2;
  uint   m_nOutputs = 2;
  uint[] m_nLayers = [3,4,5]; // [2][5,4,3][3,4,5]
  double m_learnRate = 0.5;
  double alpha = 0.1;

  const double[] WEIGHT_RANGE = [-0.1, 0.1];

  @property auto m_inputs()
    { assert(m_layers.length > 0); return m_layers[0]; }
  @property auto m_inputs(double[] p_in)
    { return m_layers[0] = p_in.dup ;}
  @property auto m_outputs()
    { assert(m_layers.length > 1); return m_layers[$-1]; }
  @property auto m_outputs(double[] p_in)
    { assert(m_layers.length > 1); return m_layers[$-1] = p_in; }
  @property auto m_hiddens()
    { assert(m_layers.length > 2); return m_layers[1..$-1]; }
  @property auto m_hiddens(double[][] p_in)
    { assert(m_layers.length > 2); return m_layers[1..$-1] = p_in; }

public:

  this() {
    m_layers.length = 0;
    m_layers.length++;
    m_layers[$-1].length = m_nInputs;
    foreach(a; m_nLayers) {
      m_layers.length++;
      m_layers[$-1].length = a;
    }
    m_layers.length++;
    m_layers[$-1].length = m_nOutputs;

    m_weights.length = 1 + m_nLayers.length;
    m_weightDelta.length = 1 + m_nLayers.length;
    foreach(i, ref a; m_weights) {
      a.length = m_layers[i].length * m_layers[i+1].length;
      m_weightDelta[i].length = m_layers[i].length * m_layers[i+1].length;
    }
    m_biases.length = m_weights.length;
    m_biasDelta.length = m_weights.length;
  }

  void randWeights() {
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

  void initDeltas(){
    foreach(ref a; m_weightDelta)
      foreach(ref b; a) b = 0.0;
    foreach(ref a; m_biasDelta) a = 0.0;
  }

  void reInit(){
    wSet = iSet = fSet = false;
    foreach(ref a; m_layers)
      foreach(ref b; a)
        b = double.nan;
    foreach(ref a; m_weights)
      foreach(ref b; a)
        b = double.nan;
    initDeltas;
  }

  bool setInput(in double[] p_in) {
    if(p_in.length != m_nInputs) return false;
    m_inputs = p_in.dup;
    iSet = true;
    fSet = false;
    return true;
  }

  bool setWeights(double[][] p_weights,double[] p_bWeights) {
    /* Verify Input */
    if(p_weights.length != m_weights.length) return false;
    if(p_bWeights.length != m_weights.length) return false;
    foreach(i, a; m_weights)
      if(a.length != p_weights[i].length) return false;
    m_weights = p_weights.dup;
    m_biases = p_bWeights.dup;
    initDeltas;
    wSet = true;
    fSet = false;
    return true;
  }

  bool setTargets(double[] p_in) {
    if(p_in.length != m_nOutputs) return false;
    m_targets = p_in.dup;
    return true;
  }

  bool feedForward() {
    if(!wSet || !iSet || fSet) return false;

    /* Sigmoid Lambda */
    auto sigmoid = (double a) => 1/(1+E^^(-a));

    /* Cycle Through each layer n+1 */
    foreach(n, ref a; m_layers[1..$])
      foreach(y, ref b; a) { /* y/y b node with each c node */
        b = 0.0;
        foreach(x, c; m_layers[n])
          b += c*m_weights[n][y*m_layers[n].length+x];
        b+= m_biases[n];
        b = sigmoid(b);
      }

    fSet = true;
    return true;
  }

  auto calcError() {
    FPTemporary!double totalError = 0.0; // Init TE
    foreach(i, a; m_outputs)
      totalError += (1.0/2.0) * (m_targets[i]-a)^^2;
    return totalError;
  }

  bool backProp() {
    if(!fSet) return false;

    FPTemporary!double eTotal = calcError;
    FPTemporary!double[][] weightDelta;
    weightDelta.length = m_weights.length;
    foreach(i, a; m_weights)
      weightDelta[i].length = a.length;
    FPTemporary!double[] biasDelta;
    biasDelta.length = m_biases.length;
    FPTemporary!double[] deltaNew;
    FPTemporary!double[] deltaOld;
    FPTemporary!double biasOld;

    //l = w % l.length
    //r = w / l.length
    /* Process the Layers */
    foreach_reverse(n, a; m_weights){

      if(n == m_weights.length-1) {
        /* jk iteration */
        foreach(x, b; a) {
          FPTemporary!double delta =
            m_layers[n+1][x/m_layers[n].length]*
            (1-m_layers[n+1][x/m_layers[n].length])*
            (m_targets[x/m_layers[n].length]-
            m_layers[n+1][x/m_layers[n].length]);
          deltaOld ~= delta;
          FPTemporary!double oJay = m_layers[n][x%m_layers[n].length];
          weightDelta[n][x] = m_learnRate*delta*oJay;
        }

        /* Do the same for the biases */
        FPTemporary!double[] deltas;
        foreach(i, o; m_layers[n+1]) deltas ~= o*(1-o)*(m_targets[i]-o);
        FPTemporary!double delta = 0.0;
        foreach(d; deltas) delta += d;
        delta /= deltas.length;
        biasOld = delta;
        FPTemporary!double oJay = 1.0;
        biasDelta[n] = m_learnRate*delta*oJay;

      } else {
        /* ij iterations */
        deltaNew.length = 0;
        foreach(x, b; a) {
          FPTemporary!double weightSum = 0.0;
          foreach(w; m_weights[n+1]) weightSum += w;
          FPTemporary!double delta =
            m_layers[n+1][x/m_layers[n].length]*
            (1-m_layers[n+1][x/m_layers[n].length])*
            weightSum * deltaOld[x/m_layers[n].length];
          deltaNew ~= delta;
          FPTemporary!double xi = m_layers[n][x%m_layers[n].length];
          weightDelta[n][x] = m_learnRate*delta*xi;
        }

        /* Do the same for the biases */
        FPTemporary!double weightSum = 0.0;
        foreach(w; m_weights[n+1]) weightSum += w;
        FPTemporary!double[] deltas;
        foreach(i, o; m_layers[n+1]) deltas ~= o*(1-o)*weightSum * biasOld;
        FPTemporary!double delta = 0.0;
        foreach(d; deltas) delta += d;
        delta /= deltas.length;
        biasOld = delta;
        FPTemporary!double oJay = 1.0;
        biasDelta[n] = m_learnRate*delta*oJay;

        /* End of cycle maintenance */
        deltaOld = deltaNew.dup;
      }
    }

    /* Apply deltas *//*TODO Apply Momentum */
    foreach(y, ref a; m_weights)
      foreach(x, ref b; a)
        b += weightDelta[y][x];

    foreach(y, ref a; m_biases)
      a += biasDelta[y];

    foreach(y, ref a; m_weightDelta)
      foreach(x, ref b; a) b = weightDelta[y][x];
    foreach(x, ref a; m_biasDelta) a = biasDelta[x];
    fSet = false;
    return true;
  }

  Nullable!double trainCycle() {
    Nullable!double output;
    if(!wSet || !iSet || fSet) return output;

    bool result = feedForward && backProp && feedForward;
    if(result) output = calcError;

    return output;
  }

  Tuple!(double, double) train(in ulong p_cyc) {
    Tuple!(double, double) output;
    if(!wSet || !iSet || fSet) return output;
    bool result = feedForward;
    if(!result) return output;

    output[0] = calcError;
    for(uint i = 0; i < p_cyc; ++i) result &= backProp && feedForward;
    if(result) output[1] = calcError;

    return output;
  }

  Nullable!ulong train(double p_thresh) {
    Nullable!ulong output;
    if(!wSet || !iSet || fSet) return output;
    bool result = feedForward;
    if(!result) return output;

    FPTemporary!double error = double.max;
    output =0;
    while(result && error > p_thresh) {
      result &= backProp && feedForward;
      error = calcError;
      output++;
    }

    if(!result) output.nullify;

    return output;
  }

  override string toString() {
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
    output ~= "Weight Values: " ~ to!string(m_weights) ~ "\n";
    output ~= "Bias Values  : " ~ to!string(m_biases) ~ "\n";
    output ~= "Input Values : " ~ to!string(m_inputs) ~ "\n";
    output ~= "Hidden Values: " ~ to!string(m_hiddens) ~ "\n";
    output ~= "Output Values: " ~ to!string(m_outputs) ~ "\n";
    output ~= "Target Values: " ~ to!string(m_targets) ~ "\n";
    output ~= "Weight Deltas: " ~ to!string(m_weightDelta) ~ "\n";
    output ~= "Bias Deltas  : " ~ to!string(m_biasDelta) ~ "\n";
    output ~= "Current Error: " ~ to!string(calcError) ~ "\n";
    output ~= "Weights Set  : " ~ to!string(wSet) ~ "\n";
    output ~= "Inputs  Set  : " ~ to!string(iSet) ~ "\n";
    output ~= "FedForward   : " ~ to!string(fSet) ~ "\n";
    output ~= "-----------------------------------";
   return output;
  }
}
