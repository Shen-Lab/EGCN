import numpy as np
import pre_pdb
import sys
import rosetta_param

prot_name = sys.argv[1]

#edge features


def inverse_distance_judge(coor1, coor2):
    cutoff=12
    x= abs(coor1[0]-coor2[0])
    if(x > cutoff):
        return 0.
    y= abs(coor1[1]-coor2[1])
    if(y > cutoff):
        return 0.
    z= abs(coor1[2]-coor2[2])
    if(z > cutoff):
        return 0.
    t = x**2 + y**2 + z**2
    if(t==0.):
        return 0.
    return 1/np.sqrt(t)
    

def resi_resi_dis(a, b):
    out = []
    out.append(inverse_distance_judge(a['S'], b['S']))
    out.append(inverse_distance_judge(a['S'], b['O']))
    out.append(inverse_distance_judge(a['S'], b['N']))
    out.append(inverse_distance_judge(a['O'], b['S']))
    out.append(inverse_distance_judge(a['O'], b['O']))
    out.append(inverse_distance_judge(a['O'], b['N']))
    out.append(inverse_distance_judge(a['N'], b['S']))
    out.append(inverse_distance_judge(a['N'], b['O']))
    out.append(inverse_distance_judge(a['N'], b['N']))
    if(a['resi_name']=='PRO' or b['resi_name']=='PRO'): 
        out.append(0.)
        out.append(0.)
    else:
        out.append(inverse_distance_judge(a['HN'],b['O']))
        out.append(inverse_distance_judge(a['O'], b['HN']))

    return out

def edge_features(prot_path1, prot_path2):
     resilist1 = pre_pdb.prepocess_pdb(prot_path1)
     resilist2 = pre_pdb.prepocess_pdb(prot_path2)

     r = np.zeros((500, 500, 11)) 
     
     resi1_length = min(500, len(resilist1))
     resi2_length = min(500, len(resilist2))
 
     for i in range(resi1_length):
        for j in range(resi2_length):
            r[i][j] = resi_resi_dis(resilist1[i], resilist2[j])

     return r

# edge feature generation
intra_rec = []
intra_lig = []
inter = []

'''
for n in range(1000):
     intra_rec = []
     intra_lig = []
     inter = []
     for i in range(1):
         pathr = "../3_decoys/"+prot_name+"_"+str(n+1)+"/smpr_"+str(i+1)
         pathl = "../3_decoys/"+prot_name+"_"+str(n+1)+"/smpl_"+str(i+1)
         intra_rec =edge_features(pathr, pathr)
         intra_lig =edge_features(pathl, pathl)
         inter     =edge_features(pathr, pathl)          
     print (n)
     np.save(output_path+"/intra_rec_"+str(n+1), intra_rec)
     np.save(output_path+"/intra_lig_"+str(n+1), intra_lig)
     np.save(output_path+"/inter"+str(n+1), inter)
'''

def node_features(prot_path):
     resilist = pre_pdb.prepocess_pdb(prot_path)
     resilist = pre_pdb.cal_sasa(prot_path, resilist)
     
     node = np.zeros((500, 4))
     resi_length = min(500, len(resilist))
     
     for i in range(resi_length):
        resi_name = resilist[i]['resi_name']
        x= []
        x.append(rosetta_param.rosetta_params[resi_name][0]) # NBR Radii
        x.append(rosetta_param.rosetta_params[resi_name][1]) # Charge
        x.append(resilist[i]['SASA'])
        dis = np.linalg.norm(np.subtract(resilist[i]['CA'],resilist[i]['S']))
        x.append(dis)
        node[i] = x

     return node


'''
# node feature generation
for n in range(1000):
     node_rec = []
     node_lig = []    
     for i in range(1):
         pathr = "../3_decoys/"+prot_name+"_"+str(n+1)+"/smpr_"+str(i+1)
         pathl = "../3_decoys/"+prot_name+"_"+str(n+1)+"/smpl_"+str(i+1)
         node_rec=node_features(pathr)
         node_lig=node_features(pathl)
     print (n)
     np.save(output_path+"/node_rec_"+str(n+1), node_rec)
     np.save(output_path+"/node_lig_"+str(n+1), node_lig)
    
     t1=np.load(output_path+"/intra_rec_"+str(n+1)+'.npy')
     t2=np.load(output_path+"/intra_lig_"+str(n+1)+'.npy')
     t3=np.load(output_path+"/inter"+str(n+1)+'.npy')

     t1=np.reshape(t1, (500,500,11))
     t2=np.reshape(t2, (500,500,11))
     t3=np.reshape(t3, (500,500,11))
     
     np.save(output_path+"/intra_rec_"+str(n+1), t1)
     np.save(output_path+"/intra_lig_"+str(n+1), t2)
     np.save(output_path+"/inter"+str(n+1), t3)
     
'''

pathr = "../unbound_pdb/"+prot_name+"/receptor.pdb"
pathl = "../unbound_pdb/"+prot_name+"/ligand.pdb"
intra_rec =edge_features(pathr, pathr)
intra_lig =edge_features(pathl, pathl)
inter     =edge_features(pathr, pathl)
node_rec=node_features(pathr)
node_lig=node_features(pathl)

np.save("../unbound_pdb/"+prot_name+"/intra_rec", intra_rec)
np.save("../unbound_pdb/"+prot_name+"/intra_lig", intra_lig)
np.save("../unbound_pdb/"+prot_name+"/inter", inter)
np.save("../unbound_pdb/"+prot_name+"/node_rec", node_rec)
np.save("../unbound_pdb/"+prot_name+"/node_lig", node_lig)





















