# Pulgon_tools


## Install
```
pip install -e .
```

## Usage
After installing the package with `pip` (e.g. `pip install -e .`), the command `pulgon-generate-structures` and `pulgon-detect-AxialPointGroup` will become available.

### 1.generate line group structures

#### (a) Symmetry-based approach (general)
 You need to specify the motif, generators of point groups and generalized translation group. An example:

```
pulgon-generate-structures-sym_based -m  [[3,np.pi/24,0.6],[2.2,np.pi/24,0.8]] -g ['Cn(6)','U()','sigmaV()'] -c {'T_Q':[5,3]} -s poscar.vasp
```   

-m: the Cylindrical coordinates of symcell   
-g: select from 'Cn(n)','sigmaV()','sigmaH()','U()','U_d(fid)' and 'S2n(n)'  
-c: select from {'T_Q':[Q,f]} and {'T_v':f}, 360/Q is the rotational degree along z axis, f is the translational distance along z axis.     
-s: saved file name
##### Note: No space in the list or dict.

#### (b) Chiral rolling approach (MoS2-type)

```
pulgon-generate-structures-chirality -c (10,0) -b ('Mo','S') -s POSCAR
```
-c: the chirality `(n,m)`   
-b: the symbols of atoms `(symbol1, symbol2)`  
-l: the bond length between `symbol1` and `symbol2`, `default=2.43`  
-d: the interlayer spacing, `default=1.57`  
-s: saved file name  


### 2. Symmetry detection
#### (a) detect axial point group
```
pulgon-detect-AxialPointGroup -p POSCAR -o
```
-p: POSCAR  
-t: Tolerance for atomic positions  
-g : Enable detecting point group     
-o : Enable the output of detecting process    

#### (b) detect cyclic group (generalized translational group)
```
pulgon-detect-CyclicGroup -p POSCAR -t 0.001 -o
```
-p: POSCAR  
-t: Tolerance for atomic positions    
-o : Enable the output of detecting process  


### 3. character table and irreps matrices
```
pulgon-irreps-tables -p POSCAR -q 0.0 -r
```
-p: POSCAR  
-q: Specify the `q` point, from 0 to 1.    
-t: Tolerance for atomic positions   
-r: Save the representation matrices. By default, only the character table is saved.  
-s: saved file name   


### 4. force constant correction:
```
pulgon-fcs-correction  -p POSCAR -b [False,False,True]  -x [1,1,3]
```
-p: The file of POSCAR. Default=`./POSCAR`.  
-b: The periodic boundary conduction of your structure. e.g.`--pbc [False, False, False]` correspond to cluster, `--pbc [True, Ture, False]` correspond to 2D structure.  
-x: The supercell matrix that used to calculate fcs. e.g.`--supercell_matrix [1, 1, 3]`. Default=`None`.    
-y: The path of `phonopy.yaml`. Default=`None`. If it's provided, `POSCAR` and `supercell_matrix` are not necessary.      
-f: The path of fcs. `FORCE_CONSTANTS` or `force_constants.hdf5`.  Default=`./FORCE_CONSTANTS`.   
-c: If the atomic distance beyond `cut_off`, the corresponding fcs are 0. Default=`15`.  
-n: Enable plotting the corrected phonon spectrum.   
-k: The k path of plotting phonon, e.g. `--k_path [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]`.  
-m: The available methods are 'convex_opt', 'ridge_model'. Default=`convex_opt`.  
-r: Enable recenter the structure. (atoms.positions - [0.5,0.5,0.5]) % 1.  


## Examples

examples/detect_linegroup.py:
```
python examples/detect_linegroup.py  examples/POSCAR  1e-3
```
