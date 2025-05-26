# Pulgon_tools_wip

sanity

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


### 4. generate character table
```
pulgon-generate-CharacterTable -F 1 -q 6 -r 2 -f 3 -n 5 -k "[0,0]" -s test
```

-F: choose the line group family from 1 to 13  
-q: helical group rotation number Q=q/r  
-r: helical group rotation number Q=q/r  
-a: translation period in z direction  
-f: translational group T(f)  
-n: rotational point group Cn   
-k: Brillouin zone k vector   
-s: saved "filename", save as filename.npy
  
For each family, the specific parameters required:   
line group 1: -q -r -f -n -k [k1,k2] -s  
line group 2: -a -n -k [k1] -s  
line group 3: -a -n -k [k1] -s  
line group 4: -a -n -k [k1, k2] -s  
line group 5: -q -r -f -n -k [k1,k2] -s  
line group 6: -a -n -k [k1] -s   
line group 7: -a -n -k [k1] -s      
line group 8: -a -n -k [k1,k2] -s     
line group 9: -a -n -k [k1] -s   
line group 10: -a -n -k [k1] -s   
line group 11: -a -n -k [k1] -s   
line group 12: -a -n -k [k1] -s   
line group 13: -a -n -k [k1,k2] -s   
