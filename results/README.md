### SKA1-low analysis results and outputs

##### A Note on file names
File names for the plots, configuration files and other data products are 
prefixed with the name of the telescope model they are associated. These are:

- `ska1_v5` for the reference SKA1-Low v5 configuration
- `model01_r<N>` for the set of models replacing SKA1-Low spiral arm clusters 
with stations arranged along the log spiral arms, where `<N>` where the number 
of clusters replaced is `N+1` starting from the centre of the telescope.
- `model02_r<N>` for the set of models replacing SKA1-Low spiral arm clusters 
with stations arranged in circles at the cluster radius, where `<N>` where the
number of clusters replaced is `N+1` starting from the centre of the telescope.
- `model03_r<N>` is a hybrid of `model03_r<N>` in the inner part of the 
telescope and `model02_r<N>` for the outer part.

##### Sub-folders
- `iantconfig/`: Layout files for use with `iantconfig` for the 
SKA1-low v5 configuration as well as the alternatives considered in this 
analysis. 

- `antenna_layout_plots/`: Plots of the antenna layouts for each of the models
being considered. 

- `uv_grid_image_0h/`: Plots of the snapshot zenith uv-coverage.

- `cable_lengths/`: Cable lengths for connecting stations to cluster centres.
Both debugging plots and txt data files of unwrap (cluster replacement) radius
vs cable length. Currently cluster cable lengths calculated using worse
case simplistic matching.
