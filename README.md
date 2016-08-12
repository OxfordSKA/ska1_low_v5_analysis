### SKA1-low v5 analysis scripts
This repo contains a set of scripts and results from a brief study looking in 
the SKA1-low v5 configuration. We attempt this by comparing the reference
v5 configuration to a set of proposed alternatives.

#### Results sub-folder
Folder containing some selected outputs, from the study including telescope
configuration files in `iantconfig` format. See the `README.md` file in the 
results folder for further details.
    
#### Scripts sub-folder
Folder containing various python scripts and modules used in the analysis. 
See the `README.md` file in the scripts folder for further details. 

#### Alternative telescope models considered
(This should be considered an initial set which we intend to iterate upon)

1. `model01_r<N>`: Unwraps (replaces) spiral arm clusters as a function of radius
 up to 6400m, replacing clusters with stations placed along the existing log 
 spiral. `<N>` in the model name is a integer in the range `(0, 8)` which
 describes the radius up to which the unwrap has been carried out; `N = 0` 
 consists of just first ring of clusters unwrapped and `N = 8` has all 9 
 clusters unwrapped. 
2. `model02_r<N>`: Unwraps (replaces) spiral arm clusters as a function of 
 radius up to 6400m, replacing clusters with stations placed in circles at the 
 radius of the cluster centre. As for `model01_r<N>`, `<N>` in the model name is 
 a integer in the range `(0, 8)` which describes the radius up to which the 
 unwrap has been carried out; `<N> = 0` consists of just first ring of clusters
 unwrapped and `<N> = 8` has all 9 clusters unwrapped.
3. `model03_r<N>`: This is a hybrid of `model01_r<N>` and `model02_r<N>` using
`model02_r<N>` up to and including the 5th set of spiral clusters and 
`model01_r<N>` for clusters at larger radii.

#### Metrics & analysis plots (current list)
- Cable length increase from the reference v5 layout.
- PSFRMS
- Azimuthally averaged PSF profile.
- (UVGAP)
- UV histogram (brightness sensitivity)
- UV coverage
- PSF images

#### Notes / TODO
- Cable length metric might need optimising slightly.
- UVGAP Python (callable) code will be needed if we want to expand the number 
of telescopes and not go via `iantconfig` manually.
- Currently PSF analysis is done with natural (and/or) uniform weighting. 
  Hopefully this is sufficient.

  




   

