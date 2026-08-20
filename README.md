# One-Class Wrapped Gaussian EEG classifier vs One-Class Riemannian Minimum Distance to the Mean

----

Implementation of the One-Class Wrapped Gaussian algorithm presented in [this paper](https://inria.hal.science/hal-05678185/file/Final_Graz2026___One_class_Wrapped_Gaussian.pdf).

Please cite as:
Marissens Cueva, V., de Surrel, T., Bougrain, L., Bidgoli, S. J., Cheron, G. M., Alvarez, A. M. C., Meistelman, C., Lotte, F., Yger, F. & Rimbert, S. (2026). Wrapped One-Class Riemannian EEG classifier for BCI-based detection of anesthetic states. In 10th International Graz Brain-Computer Interface Conference.

----

## Authors

- [Valérie Marissens Cueva](https://vmarissenscueva.github.io/)*, Inria Center at Univ. Bordeaux / LaBRI, Talence, France & Université de Lorraine, CNRS, LORIA, Nancy, France
- [Thibault de Surrel](https://thibaultdesurrel.github.io/)*, LAMSADE, CNRS, PSL Univ. Paris-Dauphine, France
- [Laurent Bougrain](https://members.loria.fr/lbougrain/), Université de Lorraine, CNRS, LORIA, Nancy, France & Sorbonne Université, ICM, CNRS, Inria, Inserm, Paris, France
- Seyed Javad Bidgoli, CHU Brugmann, Bruxelles, Belgium
- Guy Cheron, Laboratory of Neurophysiology and Movement Biomechanics, Université Libre de Bruxelles, Bruxelles, Belgium
- Ana Maria Cebolla Alvarez, Laboratory of Neurophysiology and Movement Biomechanics, Université Libre de Bruxelles, Bruxelles, Belgium
- Claude Meistelman, DevAH, University of Lorraine, Nancy, France
- [Fabien Lotte](https://sites.google.com/site/fabienlotte/), Inria Center at Univ. Bordeaux / LaBRI, Talence, France
- Florian Yger, LITIS, INSA Rouen-Normandy, France
- [Sébastien Rimbert](https://members.loria.fr/SRimbert/), Inria Center at Univ. Bordeaux / LaBRI, Talence, France

*These authors contributed equally to the paper

## Parameters

| Parameter | Default | Description                                                                                             |
|-----------|---------|---------------------------------------------------------------------------------------------------------|
| `eps`     | `0.95`  | Controls the threshold delimiting the target class as a percentage of log-likelihoods of training data. |
| `cov_diag` | `True` | Whether the dispersion matrix is only diagonal or not. Increases the complexity and computational time. |
