import numpy as np
import pre_pdb
import sys
import rosetta_param
import os

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

def main_edge(input_files, output_path):
        intra_rec = []
        intra_lig = []
        inter = []
        for i in range(int(len(input_files)/2)):
          pathr = input_files[i*2]
          pathl = input_files[i*2+1]
          intra_rec =edge_features(pathr, pathr)
          intra_lig =edge_features(pathl, pathl)
          inter     =edge_features(pathr, pathl)          
          np.save(output_path+"/intra_rec_"+str(i+1), intra_rec)
          np.save(output_path+"/intra_lig_"+str(i+1), intra_lig)
          np.save(output_path+"/inter"+str(i+1), inter)


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

     return node, len(resilist)

# node feature generation
def unbound_node(pidx, didx, prot_name, output_path):
 for n in range(pidx):
     node_rec = []
     node_lig = []    
     for i in range(didx):
         pathr = "../3_decoys/"+prot_name+"_"+str(n+1)+"/smpr_"+str(i+1)
         pathl = "../3_decoys/"+prot_name+"_"+str(n+1)+"/smpl_"+str(i+1)
         node_rec[0]=node_features(pathr)
         node_lig[0]=node_features(pathl)
         print (n)
         np.save(output_path+"/node_rec_"+str(n*didx+i+1), node_rec)
         np.save(output_path+"/node_lig_"+str(n*didx+i+1), node_lig)
     '''
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

def encounter_node(input_files, output_path):
   
     
     for i in range(int(len(input_files)/2)):
         pathr = input_files[i*2]
         pathl = input_files[i*2+1]
         pathc = "./complexsssss"
         os.system("cat "+pathr+' '+pathl+' > '+pathc)
         os.system("sed -i s/HSE/HIS/g "+pathc)
         
         node_rec, rec_num=node_features(pathr)
         node_lig, lig_num=node_features(pathl)
         resilist = pre_pdb.prepocess_pdb(pathc)
         resilist = pre_pdb.cal_sasa(pathc, resilist)
         os.system("rm "+pathc)        
 
         for j in range(rec_num):
            node_rec[j][2] = resilist[j]['SASA']
         for j in range(lig_num):
            node_lig[j][2] = resilist[j+rec_num]['SASA']

         print ('node', i+1)    
         np.save(output_path+"/node_rec_"+str(i+1), node_rec)
         np.save(output_path+"/node_lig_"+str(i+1), node_lig)
   

def main():
   #prot_name = sys.argv[1]
   input_file_path = sys.argv[1]
   output_path =sys.argv[2]
   #pidx = int(sys.argv[3])
   #didx = int(sys.argv[4])
   input_files = np.loadtxt(input_file_path, dtype='str')
   if len(input_files)%2==1:
      raise ValueError("Input_file_path error: The number of input files must be even (receptor and ligand).")
   print (input_files)
   encounter_node(input_files, output_path)
   main_edge(input_files, output_path)

'''
def capri_set_gen():
   list1 = np.loadtxt("../statistic/flexible_list", dtype='str')
   list1_num = np.loadtxt("../statistic/flexible_list_num")
   
   for i in range(len(list1)):
      encounter_node(1, list1_num[i], list1[i], "../7_resi_final/"+list1[i])  	
'''



if __name__ == "__main__":
    main()


