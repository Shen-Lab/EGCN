rosetta_params={}

rosetta_params['ALA'] = [3.245] #1
rosetta_params['ARG'] = [5.640] #2
rosetta_params['ASN'] = [4.290] #3
rosetta_params['ASP'] = [4.250] #4
rosetta_params['CYS'] = [4.170] #5
rosetta_params['GLN'] = [4.950] #6
rosetta_params['GLU'] = [4.900] #7
rosetta_params['GLY'] = [3.025] #8
rosetta_params['HSE'] = [4.660] #9
rosetta_params['ILE'] = [4.320] #10
rosetta_params['LEU'] = [4.760] #11
rosetta_params['LYS'] = [5.300] #12
rosetta_params['MET'] = [4.950] #13
rosetta_params['PHE'] = [4.815] #14
rosetta_params['PRO'] = [3.580] #15
rosetta_params['SER'] = [3.560] #16
rosetta_params['THR'] = [3.790] #17
rosetta_params['TRP'] = [4.910] #18
rosetta_params['TYR'] = [4.715] #19
rosetta_params['VAL'] = [3.900] #20

for residue in rosetta_params:
    t = open("/scratch/user/cyppsp/project_CAPRIspecial2019/data/6_resi_feat_gen/Rosetta_param/standard_aa/"+residue+".params")
    charge = 0.
    for line in t:
       if(line[0:4]=='ATOM'):
           atom = line.strip('\n').split()[1]
           if(atom != 'N' and atom != 'CA' and atom !='C' and atom!='O' and atom!='H'):
                 charge+=float(line.strip('\n').split()[4])
    rosetta_params[residue].append(charge)



