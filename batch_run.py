# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:18:08 2024

@author: user
"""

import os
import re

# grid search
if __name__ == '__main__':
    for dataset in ['gowalla']:
        with open('performance.txt','a') as fp:
            fp.write('%s\n'%dataset)
        for alpha in [0.,0.1,0.2,0.3,0.4,0.5,2.0,2.1,2.2,2.3,2.4,2.5]:
            with open('performance.txt','a') as fp:
                fp.write('When alpha==%f'%alpha)
            os.system("python update_conf.py -dataset %s -alpha %f"%(dataset, alpha))
            os.system("python main.py")
        
        
        # for step in [2,5,10,40,50,100]:
        #     with open('performance.txt','a') as fp:
        #         fp.write('When step=%d:'%step)
        #     os.system("python update_conf.py -dataset %s -step %d"%(dataset,step))
        #     os.system("python main.py")
        # for scale in [1e-5,1e-4,1e-3,1e-2,1e-1]:
        #     with open('performance.txt','a') as fp:
        #         fp.write('When scale=%f:'%scale)
        #     os.system("python update_conf.py -dataset %s -scale %f"%(dataset,scale))
        #     os.system("python main.py")
        # for min_max in [[5e-4,5e-3],[5e-4,1e-2],[1e-3,5e-3],[1e-3,1e-2],[5e-3,1e-2]]:
        #     with open('performance.txt','a') as fp:
        #         fp.write('When noise_min=%f,noise_max=%f:'%(min_max[0],min_max[1]))
        #     os.system("python update_conf.py -dataset %s -noise_min %f -noise_max %f"%(dataset,min_max[0],min_max[1]))
        #     os.system("python main.py")
        # for temp in [0.005,0.01,0.05,0.1,0.5,1.0,5.0,10.0, 15.0]:
        # # for temp in [15.,20.,25.,30.,35.,40.,45.,50.]:
        #     with open('performance.txt','a') as fp:
        #         fp.write('When temp=%f:'%(temp))
        #     os.system("python update_conf.py -dataset %s -temp %f"%(dataset,temp))
        #     os.system("python main.py")
            
# # greedy search
# datasets = ['yelp2018', 'gowalla']
# hyper_dict = {'gamma': [0.005, 0.01, 0.05, 0.1], 'alpha': [0.2, 0.5, 1.0, 2.0, 5.0, 10.0], \
#               'beta': [1.0, 5.0, 10.0, 20.0], 'tau': [0.1, 0.3, 0.5, 0.7, 0.9]}

# if __name__ == '__main__':
#     for dataset in datasets:
#         with open('performance.txt', 'a') as fp:
#             fp.write("%s\n"%dataset)
#         base = "python update_conf.py -dataset %s "%dataset
#         hyper_names = list(hyper_dict.keys())
#         final_hyper_dict = {'gamma':0.005, 'alpha':0.2, 'beta':1.0, 'tau':0.1}
#         best_performance = 0.0
#         for i in range(len(hyper_names)):
#             partial = base + "-%s %f -%s %f -%s %f "%(hyper_names[1], final_hyper_dict[hyper_names[1]], \
#                                                       hyper_names[2], final_hyper_dict[hyper_names[2]], \
#                                                       hyper_names[3], final_hyper_dict[hyper_names[3]]) 
#             for option in hyper_dict[hyper_names[0]]:
#                 with open('performance.txt', 'a') as fp:
#                     fp.write('When %s=%f:'%(hyper_names[0],option))

#                 command = partial + "-%s %f"%(hyper_names[0], option)
#                 os.system(command)
#                 os.system("python main.py")
                
#                 with open('performance.txt', 'r') as fp:
#                     l = fp.readlines()
#                 res = re.split('{|}', l[-1])[1]
#                 res = re.split(':|,', res)[1]
#                 if float(res)>best_performance:
#                     final_hyper_dict[hyper_names[0]] = option
#                     best_performance = float(res)
                    
#             with open('performance.txt', 'a') as fp:
#                 fp.write("Best option: %s"%str(final_hyper_dict))
#             hyper_names = hyper_names[1:] + [hyper_names[0]]
                        
            