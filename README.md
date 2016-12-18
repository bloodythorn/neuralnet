# neuralnet

A neural network implemented in D using Back Propagation and Momentum.

## Building

  To build you'll need a recent copy of dmd and dub. It doesn't require anything
  fancy. You don't even have to use my dub files.

  This will likely need accommodation to build on a non-linux system. I switched
  between windows and linux while building it and was constantly having to change
  ulong/uints around because of different implementations.

## Use

  Define an object with dimensions in the constructor.

  ``` NeuralNet net = new NeuralNet([5, 3, 2]); ```

  ... will define a neural net with five inputs, 2 outputs, and one hidden layer,
  three units wide. There is also the default constructor, that will
  automatically set the net for a [2, 2, 2] net. Once a net is initialized
  it currently has no way of being re-sized. You should just destroy it and
  create a new one. Minimum size would be default [2,2,2].

  The constructor handles all the internal linking, and readying the net for
  setup. Next you'll need to set your weights. For this you have two options

  1. NeuralNet.randWeights.

  ``` net.randWeights ```

  ... on the above constructed neural net will generate two rows of weights,
  one with 5 * 3 units, the second with 3 * 2 units.

  2. Flat array, set with NeuralNet.setWeights(double[]).

  This function simply takes a flat array that has enough weights in it to fill
  each allocated weight node. If there are any more, or less, the weights will
  be rejected. So for above's example, you'd need a flat array with
  (5*3+3*2) = 21 doubles.

  After that in the same manner you'll set your input and target values.

  Once that is done you are now able to run all training cycle commands from
  running individual feed forwards and back propagations, to using the single
  NeuralNet.trainCycle macro, to the cycle or threshold based NeuralNet.train
  macros.

  Currently an example is setup similar to what was used in our class project.
  It will take a bunch of 8x9 pixel images of digits 0-4 and load them in. It will train on these until the given threshold is reached. Once that is done
  it will run each training sample back through and validate whether or not
  it will get the expected result within the given threshold.

  This is example will load previously saved weights if they exist (as they
  will always be relevant to the example), and save them after each
  successful training session.

## Story

I had to make a neural net as a school project. My AI partner and I each coded
our own in tandem. The turn-in project was his, with my input file parsing. But
he had did most of the work well before turn-in and I felt I was obligated to
catch up just in case he needed help.

What you are seeing is the result of that catch-up effort. I was learning D and
it sounded like a wonderful way to sharpen my skills with it. I will continue
to develop it and eventually intend to implement it in a game.