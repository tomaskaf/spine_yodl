# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 18:44:44 2020

@author: filip
"""
#%%
import time
import sys
from convert_to_zarr1 import convert
from predict_mod_devFT import predict
from segment import evaluate_affs
import zarr
from glancer_mod import glanc
import webbrowser
import matplotlib.pyplot as plt
#%%

def spine_yodlman(container_name,
                 path_to_processed,
                 sample_name,
                 path_to_data
                 ):

    #timing
    start_time = time.time()
    
#tiffimport to zarr
   
    
    #container_name=container_name
    #path_to_processed=path_to_processed
    path_to_dataGP=path_to_data+'/'+container_name+'.zarr'
   
    
    #sample_name=sample_name
    
    convert(sample_name, 'validate', 'sample1', path_to_processed, container_name)
    

    path_to_output='C:/Users/filip/spine_yodl/validate/'+ container_name+'.zarr'

    
    #running the gp pipeline
    
    
    iteration = 100000
    thresholds = [50]
    
    raw, affs_predicted = predict(iteration,path_to_dataGP)
    plt.close()
    f = zarr.open(path_to_output)
    f[f'{sample_name}/raw'] = raw
    f[f'{sample_name}/affs_predicted'] = affs_predicted
    
    labels = zarr.open(path_to_dataGP+'/source/'+sample_name + '/labels')[:]
    
    segmentations, scores, fragments = evaluate_affs(
        affs_predicted,
        labels,
        thresholds=thresholds)
    
    # print(scores)
    
    # segmentations, fragments = segment_affs(
    #     affs_predicted,
    # #     labels,
    #      thresholds=thresholds)
  
    
    f[f'{sample_name}/segmentation'] = segmentations[0]
    f[f'{sample_name}/fragments'] = fragments[0]
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
    url=glanc(container_name,sample_name,path_to_output)
    #time.sleep(5)
    # chrome_path='C:\Program Files (x86)\Google\Chrome\Application\chrome.exe %s'   
    # webbrowser.get(chrome_path).open(url)
    # input("Press Enter to continue...")
    print(url)
    
    path_to_tif=(path_to_output+'/'+sample_name+'_predicted'+'.tif')
    
    return(path_to_tif,scores,segmentations)
#%%    
if __name__ == "__main__":
    container_name =  str(sys.argv[1])
    path_to_processed = str(sys.argv[2])
    sample_name = str(sys.argv[3])
    path_to_data = str(sys.argv[4])
  #     # container_name = '20200406'
  #     path_to_processed = 'C:/Users/filip/spine_yodl/data'
  #     sample_name = 'ref_PRE__00001_REF_Ch1_0926B'   

  #     path_to_tif = spine_yodlman('20200407_mtlb','C:/Users/filip/spine_yodl/data','refPRE__00001_REF_Ch1_LA1102295')

  #     print(path_to_tif)
  # ('20200406_mtlb','C:/Users/filip/spine_yodl/data','ref_PRE__00001_REF_Ch1_0926B')
    sys.stdout.write(str(spine_yodlman(container_name,path_to_processed, sample_name, path_to_data)))
   
    
    

    #%%
    
    
# if __name__ == "__main__":

#      container_name = '20200406'
#      path_to_processed = 'C:/Users/filip/spine_yodl/data'
#      sample_name = 'ref_PRE__00001_REF_Ch1_0926B'      
                                                                
#      spine_yodlman(container_name,path_to_processed,sample_name)
        
