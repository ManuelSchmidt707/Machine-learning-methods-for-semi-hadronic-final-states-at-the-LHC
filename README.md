# Machine learning methods for (semi-) hadronic final states at the LHC

We present here the neural networks used in the thesis "Machine learning methods for (semi-) hadronic final states at the LHC".
This work was done at the University of Würzburg under the supervision of Prof. Dr. W. Porod.

We analyze simulated LHC events that produce a final state with high hadronic activity and two isolated same-sign leptons. This process is motivated by the search for physics beyond the standard model. Composite Higgs models, a theoretically well motivated extensions of the standard model, predict new doubly charged scalar particles with this signature.
We have used three different representations of the LHC events, namely jet images, particle clouds and a set of high-level kinematic data, and use deep neural networks to discriminate the signal process from relevant background processes.
We target each representation with specialized neural networks and present those in this repository. Finally, we combine these networks into a single classifier, to produce a combined classification.

The networks are separated by the type of representation they work on.

**The default arguments of the network classes represent the settings used in the thesis.**
