# Filename: reshape_example.py
# Function: numpy reshape example

import numpy as geek 

aantal_matrices = 5
aantal_rijen = 2
aantal_kolommen = 3

array = geek.arange(aantal_matrices*aantal_rijen*aantal_kolommen) # creeer een matrix met 1 rij met n kolommen
print("Original array : \n", array) 

out = geek.reshape(array, (aantal_matrices,aantal_rijen,aantal_kolommen))
print("\narray reshaped: \n", out) 
  