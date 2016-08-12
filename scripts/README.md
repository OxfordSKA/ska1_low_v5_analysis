### SKA1-low analysis scripts

This folder contains scripts used to analyse various 
alternative SKA1-Low v5 configurations.

#### Scripts
- `analyse_v5.py` is an experimental script iterating over clusters in the 
spiral arms of the v5 reference design, unwrapping them in various ways and 
producing a set of metrics from set of configurations this produces.  


#### Utilities module
The `utilities` is a python package which contains 3 main python classes.

- `Layout`: Class containing various coordinate generators used in building 
telescope configurations.
- `Telescope`: Class which holds a model of a telescope and methods for 
building the model from various components (random core, spiral arms etc.).
This class is also has the ability to load the current v5 reference design 
with a radius filter.
- `Analysis`: Inherits the `Telescope` class adding various metrics which 
can be used to analyse the configurations.



