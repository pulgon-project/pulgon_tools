# Pulgon_tools



## Usage
After installing the package with `pip` (e.g. `pip install -e .`), the command `pulgon-generate-structures` and `pulgon-detect-AxialPointGroup` will become available.

### 1.generate line group structures

(a) Symmetry-based approach (general)
 You need to specify the motif, generators of point groups and generalized translation group. An example:

```
pulgon-generate-structures-sym_based -m  [[3,np.pi/24,0.6],[2.2,np.pi/24,0.8]] -g ['Cn(6)','U()','sigmaV()'] -c {'T_Q':[5,3]} -s poscar.vasp
```   

-m: the Cylindrical coordinates of symcell   
-g: select from 'Cn(n)','sigmaV()','sigmaH()','U()','U_d(fid)' and 'S2n(n)'  
-c: select from {'T_Q':[Q,f]} and {'T_v':f}, 360/Q is the rotational degree along z axis, f is the translational distance along z axis.     
-s: saved file name  

##### Note: No space in the list or dict.

(b) Chiral rolling approach (MoS2-type)

```
pulgon-generate-structures-chirality -c (10,0) -sy (Mo,S) -s POSCAR
```   
-c: the chirality (n,m)   
-sy: the symbols of atoms (symbol1, symbol2)  
-b: the bond length between `symbol1` and `symbol2`, `default=2.43`  
-dz: the interlayer spacing, `default=1.57`  
-s: saved file name  


### 2. detect axial point group
```
pulgon-detect-AxialPointGroup poscar.vasp --enable_pg --ebable_log
```

--enable_pg : Enable detecting point group   
--ebable_log : Enable the output of detecting process   

### 3. detect cyclic group (generalized translational group)
```
pulgon-detect-CyclicGroup poscar.vasp --ebable_log
```
--ebable_log : Enable the output of detecting process


### 4. character table
```
pulgon-irreps-tables poscar --qpoint_z 0.0 --enable_rep_matrix
```
--qpoint_z: Specify the `q` point.  
--enable_rep_matrix: Save the representation matrices. By default, only the character table is saved.


### 5. force constant correction:
```
pulgon-fcs-correction --pbc [True,True,False] --poscar POSCAR --supercell_matrix [7,7,1] --recenter
```
`--pbc`: The periodic boundary conduction of your structure. e.g.`--pbc [False, False, False]` correspond to cluster, `--pbc [True, Ture, False]` correspond to 2D structure.  
`--poscar`: The file of POSCAR. Default=`./POSCAR`.    
`--supercell_matrix`: The supercell matrix that used to calculate fcs. e.g.`--supercell_matrix [5, 5, 1]`. Default=`None`.    
`--path_yaml`: The path of `phonopy.yaml`. Default=`None`. If it's provided, `POSCAR` and `supercell_matrix` are not necessary.      
`--fcs`: The path of fcs. `FORCE_CONSTANTS` or `force_constants.hdf5`.  Default=`./FORCE_CONSTANTS`.   
`--cut_off`: If the atomic distance beyond `cut_off`, the corresponding fcs are 0. Default=`15`.  
`--recenter`: Enable recenter the structure. (atoms.positions - [0.5,0.5,0.5]) % 1.  
`--plot_phonon`: Enable plotting the corrected phonon spectrum.   
`--k_path`: The k path of plotting phonon, e.g. `--k_path [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]`.  
`--phononfig_savename`: The name of phonon spectrum fig. Default=`phonon_fix.png`.   
`--fcs_savename`: The name of saving corrected fcs file. Default=`FORCE_CONSTANTS_correction.hdf5`.   
`--full_fcs`: Enable saving the complete fcs.   
`--methods`: The available methods are 'convex_opt', 'ridge_model'. Default=`convex_opt`.  


## Examples

examples/detect_linegroup.py:
```
python examples/detect_linegroup.py  examples/POSCAR  1e-3
```
