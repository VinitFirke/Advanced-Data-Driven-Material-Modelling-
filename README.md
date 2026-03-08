# Advanced-Data-Driven-Material-Modelling-

This repo contains my final version of Inverse Problem for material parameter estimation using Proper Orthogonal Decomposition technique

-----------------

This repository focuses on reduced-order modeling for identifying optimized material parameters from the full-field experimental data by considering linear elasticity theory into account. The parameters - Shear Modulus (G) and Bulk Modulus (K) - vectorial notation used as $\boldsymbol{\kappa}$ are first identified using the Finite Element Method in FeniCSx as the Full-Order Model (FOM) and later on Proper Orthogonal Decomposition is employed as a model order reduction approach. This reduced order model (ROM) is first benchmarked with the full order model, later on compared with the experimental data, and finally its effectiveness to be used as a surrogate to the FOM is explored.

## Identification from Full-Field Data

The objective of this project is to identify the material parameters $\boldsymbol{\kappa}$ from full-field displacement data.

The system is governed by the state equation

$$
\mathbf{F}(\mathbf{y}; \boldsymbol{\kappa}) = \mathbf{0},
$$

where $\mathbf{y}$ represents the state variable and $\boldsymbol{\kappa}$ denotes the material parameters.

The parameter identification problem is formulated as the minimization of the mismatch between simulated and measured data:

$$
\boldsymbol{\kappa}^* = \arg\min_{\kappa} \left\| \mathbf{O}(\mathbf{y}(\boldsymbol{\kappa})) - \mathbf{d} \right\|_2^2.
$$

### Simplifications

To reduce computational cost, the following simplifications are used:

- **Fixed observation operator:**  
  The observation operator $\mathbf{O}$ only selects the FEM nodes contributing to the loss. It does not depend on the state or the material parameters.

- **One-time interpolation of experimental data:**  
  The experimental displacement data is interpolated once from the sensor locations onto the FEM grid, resulting in $\tilde{\mathbf{d}}$. This avoids repeated interpolation during every optimization step.

With these modifications, the final identification problem becomes

$$
\boldsymbol{\kappa}^* = \arg\min_{\kappa} \left\| \mathbf{O}(\mathbf{y}(\boldsymbol{\kappa})) - \tilde{\mathbf{d}} \right\|_2^2
= \arg\min_{\kappa} \mathcal{L}(\mathbf{y}(\boldsymbol{\kappa})).
$$

In summary, the material parameters are identified by solving the forward problem and minimizing the loss between FEM-predicted displacements and interpolated experimental full-field data.


## Specimen Geometry and Boundary Conditions

The specimen consists of a **clamped left boundary** and a **tensile load applied on the right boundary**. The experimental dataset, provided by Tröger et al., contains two types of displacement data:

- **Raw displacement data:** measured using the **Digital Image Correlation (DIC)** method in the region enclosed by the solid boundary.
- **Interpolated displacement data:** measured displacements mapped onto equidistant points in the region of interest. These data contain some outliers caused by the stiffness of the tensile testing machine, which are removed in this study.

### Specimen Geometry

![Specimen Geometry](PDF/Geometry.png)

### Experimental Displacement Data

After removing the outliers, the interpolated experimental displacement fields $u_x$ and $u_y$ are used for the parameter identification process.

[View experimental displacement plot](PDF/Experimental_Displacement_data.pdf)
