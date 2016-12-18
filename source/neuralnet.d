module neuralnet;

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

/**
  @brief Neural Net Class

  Description: This class will setup a neural net and given the proper
    information will allow for a feedforward recognition and a
    backpropagation training cycle.
  */
class NeuralNet {

  /* Structure for a Weight node */
  private struct Weight {
    double    weight    = 0;   // Current weight of unit.
    double    deltaW    = 0;   // Most recent delta before applied.
    double    deltaWOld = 0;   // Old delta for momentum.
    string    weightNum = "-"; // String Representation of location.
    Neuron*[] lefts;           // Left side (upstream) Neurons
    Neuron*[] rights;          // Right side (downstream) Neurons

    /* Function that calcs back prop */
    void fire() {
      enforce(lefts.length == 1 && rights.length == 1);
      auto Ok = rights[0]; auto Oj = lefts[0];
      deltaW = LEARN_RATE*Ok.getDelta*Oj.value;
    }

    /* Function that merges deltas and applies momentum */
    void mergeDelta() {
      weight += deltaW + MOMENTUM_ALPHA * deltaWOld;
      deltaWOld = deltaW;
      deltaW = 0;
    }

    /* Two different string reps for debugging */
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

  /* Structure for a Neuron node */
  private struct Neuron {
    bool   bias   = false;  // Is this node a bias node?
    bool   dSet   = false;  // Delta calculated flag
    double value  = 0;      // value set by feed forward (or set as input)
    double target = 0;      // set target (if output node)
    double delta  = 0;      // calculated delta
    string neuronNum = "+"; // String rep for the node
    Weight*[] lefts;        // Left side (upstream) weights
    Weight*[] rights;       // Right side (downstream) weights

    /* This function feeds the node forward */
    void fire() {
      auto ll = lefts.length; auto rl = rights.length;
      if(ll != 0) // Not a bias (biases don't get sigmoid)
        value = 1/(1+E^^(-value)); // Sigmoid
      if(rl != 0) // Feed Forward (if not an output)
        foreach(x, ref a; rights) foreach(y, ref b; a.rights)
          b.value += (bias ? 1 : value) * a.weight;
      dSet = false; delta = 0; // Reset Delta on FF
    }

    /* Calculates the error - only for outputs */
    double error() { return (1.0/2.0) * (target-value)^^2; }

    /* This will retrieve this neuron's delta */
    double getDelta() {
      if(dSet) return delta;  // Only calc delta once.
      double tVal = ((bias) ? 1.0 : value); // Bias or not
      delta = tVal*(1-tVal);  // Initial value for delta
       // We're an output, our delta is target based.
      if(rights.length == 0) delta = delta*(target-tVal);
      else delta =  // Otherwise we got some summing to do.
        delta*reduce!((a,b)=>a+b.weight*b.rights[0].getDelta)(0.0, rights);
      dSet = true;  // Once we're calced, we shouldn't have to again.
      return delta;
    }

    /* Two different string outputs for debugging */
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

  /* Internal Arrays of Data */
  private Neuron[][] m_neurons;
  private Weight[][] m_weights;

  /* States */
  bool wSet = false;
  bool iSet = false;
  bool fSet = false;
  bool tSet = false;

  /* TODO a way to externally control these. */
  private uint[] m_size = [2,2,2];    // Default size
  private static auto fmt = "%+.2g";  // Double output format.
  private static const double   MOMENTUM_ALPHA = 0.1;  // Alpha
  private static const double   LEARN_RATE = 0.3;      // Learning Rate
  private static const double[] WEIGHT_RANGE = [-1.0, 1.00]; // For randWeights

  /* Constructors */
  public this() { init; }

  public this(uint[] p_size) {
    enforce(p_size.length > 2);
    enforce(reduce!((a,b)=> a && (b > 1))(true, p_size));
    m_size = p_size.dup; init;
  }

  /**
    @brief Initializes Neural Net

    Description: This function initializes all the references between each node
      to allow feed forward and back propagation all work internally.
    */
  private void init() {

    /* Indexing Lambdas */
    auto getXNeu = (ulong pn, ulong pi)
      { return &m_neurons[pn][pi%(m_neurons[pn].length-1)]; };
    auto getYNeu = (ulong pn, ulong pi)
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

    /* Set bias on last neuron [0..n-1] */
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

  /**
    @brief Reset Neurons

    Description: This function will clear out all values stored in the neurons
    */
  private void reset(){
    fSet = false;
    foreach(ref a; m_neurons[1..$])
      foreach(ref b; a)
        b.value = 0;
    fSet = false;
  }

  /* Weight Functions */

  /**
    @brief Clear Weights

    Description: This will clear the numeric values of all the weights, unset
      the state.
    */
  public void clearWeights() {
    foreach(ref a; m_weights) foreach(ref b; a) b.weight = 0.0;
    wSet = false; fSet = false;
  }

  /**
    @brief Getter for the Weights

    Description: This function will return a flat array with all the neural
      net's weights.
    */
  public double[] getWeights() {
    double[] output;
    if(!wSet) return output;
    foreach(a; m_weights) {
      foreach(b; a) output ~= b.weight;
    }
    return output;
  }

  /**
    @breif Property for weights length

    Description: Being that you have to use a flat-array to set the weights,
      I thought it might have to have the length set public too. With this
      function you can see how long your weight array needs to be to be
      accepted
    */
  public uint getWeightsLength() {
    uint total = 0;
    foreach(n,a; m_size[0..$-1]) total += m_size[n] * m_size[n+1] +1;
    return total;
  }

  /**
    @brief Randomize Weights

    Description: This function will randomize the weights of the neural net
      within the given range of WEIGHT_RANGE.
    */
  public void randWeights() {
    Random gen;
    foreach(ref a; m_weights)
      foreach(ref b; a)
        b.weight = uniform(WEIGHT_RANGE[0], WEIGHT_RANGE[1], gen);
    wSet = true;
  }

  /**
    @brief Weights Setter

    Description: Given a flat array of size NeuralNet.getWeightsLength, this
      will set the weights to the given values.
    */
  public bool setWeights(double[] p_in) {
    uint total = 0;
    foreach(n,a; m_size[0..$-1]) total += m_size[n] * m_size[n+1] +1;
    if(p_in.length != total) return false;
    total = 0;
    foreach(y, ref a; m_weights) foreach(x, ref b; a) b.weight = p_in[total++];
    wSet = true; fSet = false;
    return true;
  }

  /**
    @brief Getter for Inputs Length

    Description: Will return the size of the input layer.
    */
  public ulong getInputsLength() { return m_neurons[0].length; }

  /**
    @brief Input Setter

    Description: This function will accept a flat array of size
      NeuralNet.getInputsLength, and will set the input to the given values.
    */
  public bool setInputs(in double[] p_in) {
    if(p_in.length != m_size[0]) return false;
    reset;
    foreach(i, ref a; m_neurons[0][0..$-1])
      a.value = p_in[i];
    iSet = true; fSet = false;
    return true;
  }

  /**
    @brief Getter for Outputs Length

    Description: Will return the size of the output layer.
    */
  public ulong getOutputsLength() { return m_neurons[$-1].length; }

  /**
    @brief Target Setter

    Description: This function will accept a flat array of size
      NeuralNet.getOutputsLength, and will set the targets to the given values.
    */
  public bool setTargets(double[] p_in) {
    if(p_in.length != m_size[$-1]) return false;
    reset;
    foreach(i, ref a; m_neurons[$-1])
      a.target = p_in[i];
    tSet = true;
    return true;
  }

  /**
    @brief Feed Forward

    Description: This function will perform the feed forward by firing each
      neuron in-order. Feed Forward will not fire unless the weights are set
      the input is set the targets are set, and the feed forward has not been
      set.
    */
  public bool feedForward() {
    if(!wSet || !iSet || !tSet || fSet) return false;

    foreach(ref a; m_neurons) foreach(ref b; a) b.fire();

    fSet = true;
    return true;
  }

  /**
    @brief Back Propagation

    Description: This function performs the backprop step by going through each
      weight row backwards, column forwards, and fires each, which will
      calculate each weight's require delta from the set target. Once that is
      complete, it will go back over each weight again, finally integrating the
      delta with the weight, while adding in momentum.
      Back Prop will not work unless the net has been fed forward.
    */
  public bool backProp() {
    if(!fSet) return false;

    /* Generate Weight Deltas */
    foreach_reverse(n, ref a; m_weights) foreach(y, ref b; a[0..$-1]) b.fire();

    /* Apply Weight Deltas to Weights */
    foreach(ref a; m_weights) foreach(ref b; a) b.mergeDelta;

    reset; fSet = false;
    return true;
  }

  /**
    @brief Calculate Current Error

    Description: This function will calculate the error, assuming the mechanism
      has been successfully fed forward.

    */
  public auto calcError() {
    FPTemporary!double totalError = 0.0; // Init TE
    if(!fSet) return totalError;
    foreach(a; m_neurons[$-1]) totalError += a.error;
    return totalError;
  }

  /**
    @brief Run Train Cycle

    Description: This function will cycle the net between backprop and feed
      forward once. It does not matter if it has or has not been fed forward,
      as long as input, targets and weights have been set
    */
  public auto trainCycle() {
    Tuple!(double, double) output;
    if(!wSet || !iSet || !tSet) return output;
    bool result;
    if(!fSet) { // Feed Forward if it hasn't already.
      result = feedForward;
      if(!result) return output;
    }
    output[0] = calcError;

    result = backProp && feedForward;
    if(!result) return output;
    output[1] = calcError;
    return output;
  }

  /**
    @brief Train for n Cycles

    Description: This function will train the net for p_cyc cycles assuming
      it has been set with inputs/weights/targets. Once done it will return
      a tuple containing the error when it started, and the error when it
      finished.
    */
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

  /**
    @brief Train to n threshold

    Description: This function, assuming the net has been setup, will train it
      until the error rate is reduced below the threshold given. It will print
      out a status every 5 seconds of training. Once done it will return a
      tuple containing the starting error, and how many epochs it took it to
      reach the desired threshold.
    */
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

  /**
    @brief toString override

    For Debugging.
    */
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