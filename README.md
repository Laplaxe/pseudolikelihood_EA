# Code of paper "Pseudo-likelihood produces associative memories able to generalize, even for asymmetric couplings"
Francesco D'Amico, Dario Bocchi, Luca Maria Del Bono, Saverio Rossi, Matteo Negri

ArXiv: https://arxiv.org/abs/2507.05147

The code released has been produced to run simulations for pseudolikelihood training of a binary, two-bodies interaction model.
This code presents the code for the treatment of the Edwards-Anderson (EA) 2D model.

## Structure of the repository
- Codice/Modern/pl_EA2d contains the code for performing pseudolikelihood inference with a bash script to perform multiple runs
- Dati contains data for reproducing the figures of the paper. In particular:
  -  Alpha/dati_PT/hard_instances_couplings/L32 contains the file(s) of the couplings of the EA model.
  -  Omega/REsults contains the results of the pseudolikelihood inference procedure
- Grafici contains the code for producing the figures and the figures themselves 

---
