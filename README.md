# Generate_line_group_structure_module



## Usage
After installing the package with `pip` (e.g. `pip install -e .`), the command `pulgon-generate-structures` and `pulgon-detect-AxialPointGroup` will become available.

### 1.generate line group structures
 You need to specify the motif, generators of point groups and generalized translation group. An example:

windows:
```
pulgon-generate-structures -m  [[3,np.pi/24,0.6],[2.2,np.pi/24,0.8]] -g ['Cn(6)','U()','sigmaV()'] -c {'T_Q':[5,3]} -s poscar.vasp
```   
Unix:
```
pulgon-generate-structures -m  "[[3,np.pi/24,0.6],[2.2,np.pi/24,0.8]]" -g "['Cn(6)','U()','sigmaV()']" -c "{'T_Q':[5,3]}" -s poscar.vasp
```   


-m: the Cylindrical coordinates of initial atom position   
-g: select from 'Cn(n)','sigmaV()','sigmaH()','U()','U_d(fid)' and 'S2n(n)'  
-c: select from {'T_Q':[5,3]} and {'T_v':3}  
-s: saved file name  

### 2. detect axial point group
```
pulgon-detect-AxialPointGroup poscar.vasp --enable_pg
```

--enable_pg : detecting point group


##### Note: No space in the list or dict.
