# CMB_Topology_ML
Neural network layers and networks to study the topology of the universe. 

The code here is based havily on deepsphere (https://github.com/deepsphere/deepsphere-cosmo-tf2) with some minor modifications and extensions. Although deepsphere is not necessary to run the code in this repo, the `SphereHealpix` class from `pygsp.graphs` is required and can be installed from "PyGSP @ git+https://github.com/jafluri/pygsp.git@sphere-graphs". 

The datasets in the [data](data) folder are simulated spherical harmonic coefficients for CMB full sky maps with E1 topology with a small fundamental domain size. The maps are generated via the code at https://github.com/LilleJohs/CMB_Topology. These files are quite large so you'll need git-lfs installed to clone this repo.

To run the training loop, do `python train.py` or `python train_v2.py`. Don't try this without any GPUs, each epoch takes about 2.5-3 minutes on 4 NVIDIA v100 GPUs and about 6-7 minutes on 2 NVIDIA p100 GPUs.
