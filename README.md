# Pulgon_tools



## Usage
After installing the package with `pip` (e.g. `pip install -e .`), the command `pulgon-generate-structures` and `pulgon-detect-AxialPointGroup` will become available.

### 1.generate line group structures
 You need to specify the motif, generators of point groups and generalized translation group. An example:

```
pulgon-generate-structures -m  [[3,np.pi/24,0.6],[2.2,np.pi/24,0.8]] -g ['Cn(6)','U()','sigmaV()'] -c {'T_Q':[5,3]} -s poscar.vasp
```   



-m: the Cylindrical coordinates of initial atom position   
-g: select from 'Cn(n)','sigmaV()','sigmaH()','U()','U_d(fid)' and 'S2n(n)'  
-c: select from {'T_Q':[5,3]} and {'T_v':3}  
-s: saved file name  

##### Note: No space in the list or dict.


### 2. detect axial point group
```
pulgon-detect-AxialPointGroup poscar.vasp --enable_pg
```

--enable_pg : detecting point group


### 3. detect cyclic group (generalized translational group)
```
pulgon-detect-CyclicGroup poscar.vasp
```


### 4. character table



### 5. force constant correction:
```
pulgon-fcs-correction --pbc [True,True,False]
```
`--pbc`: The periodic boundary conduction of your structure. e.g.`--pbc [False, False, False]` correspond to cluster, `--pbc [True, Ture, False]` correspond to 2D structure.  
`--poscar`: The file of POSCAR. Default=`./POSCAR`.    
`--supercell_matrix`: The supercell matrix that used to calculate fcs. e.g.`--supercell_matrix [5, 5, 1]`. Default=`None`.    
`--path_yaml`: The path of `phonopy.yaml`. Default=`None`, if it's provided, `POSCAR` and `supercell_matrix` are not necessary.      
`--fcs`: The path of fcs. `FORCE_CONSTANTS` or `force_constants.hdf5`.  Default=`./FORCE_CONSTANTS`.   
`--cut_off`: fcs is 0 if the atomic distance beyond `cut_off`. Default=`15`.  
`--recenter`: Enable recenter the structure. (atoms.positions - [0.5,0.5,0.5]) % 1.  
`--plot_phonon`: Enable plotting the corrected phonon spectrum.   
`--phononfig_savename`: The name of phonon spectrum fig. Default=`phonon_fix.png`.   
`--fcs_savename`: The name of saving corrected fcs file. Default=`FORCE_CONSTANTS_correction.hdf5`.   
`--full_fcs`: Enable saving the complete fcs.   
`--methods`: The available methods are 'convex_opt', 'ridge_model'. Default=`convex_opt`.  


## Examples

examples/detect_linegroup.py:
```
python examples/detect_linegroup.py  examples/POSCAR  1e-3
```
