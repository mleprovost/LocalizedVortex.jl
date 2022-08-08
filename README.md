# LocalizedVortex.jl


In this repository, we assess the performance of the stochastic ensemble Kalman filter (sEnKF) and the localized stochastic ensemble Kalman filter (LocEnKF) to assimilate pressure observations collected on the surface of an impulsively translating flat plate. These surfaces pressure observations are used to estimate the positions and strengths of a limited collection of regularized point voritices that compactly represent the flow field. The setup of the numerical experiments are described in
[Le Provost and Eldredge (2021), "Ensemble Kalman filter for vortex models of disturbed aerodynamic flows", *Physical Review Fluids*, 6(5), 050506.](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.6.050506).


This repository contains the source code and an example Jupyter notebook that compares the performance of the LocEnKF and the sEnKF. We show the limitations of the localized sEnKF to assimilate non-local observations such as pressure observations. Indeed the pressure observations constitute noisy and spatially limited evaluations of the solution of an elliptic partial differential equation, namely the pressure Poisson equation, whose right-hand side nonlinearly depends on all the point singularities. These limitations are the foundations for the low-rank ensemble Kalman filter introduced in [Le Provost, Baptista, Marzouk, and Eldredge (2022), "A low-rank ensemble Kalman filter for elliptic observations," *arXiv preprint*, arXiv:2203.05120](https://arxiv.org/abs/2203.05120).

## Installation

This package works on Julia `1.6` and above. To install from the REPL, type
e.g.,
```julia
] add https://github.com/mleprovost/LocalizedVortex.jl.git
```

Then, in any version, type
```julia
julia> using LocalizedVortex
```

## References

[^1]: Le Provost and Eldredge (2021), "Ensemble Kalman filter for vortex models of disturbed aerodynamic flows", *Physical Review Fluids*, 6(5), 050506.

[^2]: Le Provost, Baptista, Marzouk, and Eldredge (2022), "A low-rank ensemble Kalman filter for elliptic observations," *arXiv preprint*, [arXiv:2203.05120](https://arxiv.org/abs/2203.05120).


## Licence

See [LICENSE.md](https://github.com/mleprovost/LocalizedVortex.jl/raw/main/LICENSE.md)
