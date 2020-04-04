import numpy as np
import os
import freesasa

protein ='1A2K'

os.system("cat ../3_decoys/"+protein+"_1/smpr_1 ../3_decoys/"+protein+"_1/smpl_1 > ../3_decoys/"+protein+"_1/smpc_1")
os.system("sed -i 's/HSE/HIS/g' ../3_decoys/"+protein+"_1/smpc_1")

#structure = freesasa.Structure("../3_decoys/"+protein+"_1/smpc_1")
structure = freesasa.Structure("./temp-charmm19.pdb")

result = freesasa.calc(structure)
area_classes = freesasa.classifyResults(result, structure)

selections = freesasa.selectArea(('aromatic, chain a and resi 2 and name  HA', 'r1_10, resi 1-10'),
                                 structure, result)
for key in selections:
    print key, ": %.2f A2" % selections[key]
