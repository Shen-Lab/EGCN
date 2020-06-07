import numpy as np
import freesasa
import sys
#calculating the residue-based distance features between two proteins (could be the same proteins or not)

def cal_sasa(prot, resilist):
      structure = freesasa.Structure(prot)
      result = freesasa.calc(structure)

      for i in range(len(resilist)):
          resi_ind = resilist[i]['resi_seq']
          chain = resilist[i]['chain']
          sasa_value = freesasa.selectArea(('alanine, resn ala', 'we, resi '+str(resi_ind)+' and chain '+chain),
                                 structure, result)
          resilist[i]['SASA']=sasa_value['we']
      return resilist


def prepocess_pdb(prot):
      resilist=[]
      for line1 in open(prot):
           if(line1[0:3]=='ATO'):
               residue_name=line1[17:20]
               residue_seq=int(line1[22:26])
               chain_id=line1[21]
               atom_name = line1[12:16].strip(' ')
               coor =[float(line1[30:38]), float(line1[38:46]), float(line1[46:54])]
               if(resilist==[] or resilist[-1]['resi_seq'] != residue_seq  or resilist[-1]['chain'] != chain_id ):
                  new_resi = {}
                  new_resi['resi_name']=residue_name
                  new_resi['resi_seq'] =residue_seq
                  new_resi['chain']=chain_id
                  new_resi['S'] = []
                  #print (new_resi)
                  #if(resilist!=[]):
                  #  print (len(resilist[-1]['S']), resilist[-1]['resi_name'])
                  resilist.append(new_resi)
               if(atom_name == 'CA' or atom_name =='C' or atom_name =='N' or atom_name =='HN' or atom_name == 'O'):
                  resilist[-1][atom_name] = coor
               elif atom_name == 'HT1' or atom_name == 'HN1':
                  resilist[-1]['HN'] = coor
               elif atom_name == 'OT1' or atom_name == 'OT':
                  resilist[-1]['O'] = coor 
               elif atom_name[0]!='H' or (residue_name == 'GLY' and atom_name=='HA2'):
                  resilist[-1]['S'].append(coor) 
      for i in range(len(resilist)):
           resilist[i]['S']=np.mean(resilist[i]['S'], axis=0)
           # calculating SASA
  



      return resilist


if __name__=='__main__':
 pdb_path = sys.argv[1]
 resilist = prepocess_pdb(pdb_path)
 resilist = cal_sasa(pdb_path, resilist)
 print (len(resilist))
	
               

			

