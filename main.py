from FuxiRec import FuxiRec
from util.conf import ModelConf

if __name__ == '__main__':
    # Register your model here
    classic_baselines = ['c1. MF', 'c2. NeuMF']
    graph_baselines = ['g1. APPNP', 'g2. DirectAU', 'g3. LightGCN', 'g4. LTGNN', 'g5. ForwardRec', 'g6. HGCF', 'g7. LinkProp']
    graph_signal = ['gs1. LGCN', 'gs2. PGSP', 'gs3. JGCF', 'gs4. SGFCF', 'gs5. ChebyCF',]
    hypergraph = ['hg1. DHCF', 'hg2. HCCF']
    social_recommendations = ['sr1. DiffNet', 'sr2. DiffNet++', 'sr3. MHCN', 'sr4. SEPT']
    negative_sampling = ['ns1. MixGCF', 'ns2. DENS', 'ns3. DINS', 'ns4. AHNS']
    ssl_graph_models = ['sg1. SGL', 'sg2. BUIR','sg3. SSL4Rec', 'sg4. SimGCL', 'sg5. NCL', 'sg6. AdaGCL', 'sg7.SelfCF ', \
                        'sg8. LightGCL', 'sg9. XSimGCL', 'sg10. EGCF', 'sg11. SCCF', 'sg12. RecDCL', 'sg13. SGCL']
    sequential_baselines= ['s1. SASRec', 's2. FBABRF']
    ssl_sequential_models = ['ss1. CL4SRec','ss2. DuoRec','ss3. BERT4Rec']
    autoencoder_baselines = ['a1. CDAE', 'a2. MultiVAE']
    diffusion_models = ['d1. CODIGEM', 'd2. DiffRec', 'd3. L-DiffRec', 'd4. BSPM', 'd5. GiffCF', 'd6. DDRM', 'd7. PreferDiff-G', 'd8. HDRM', 'd9. TV-Diff',]
    test_models = ['1. test1', '2. CoRec', '3. DiffGraph',  '4. GumbelDiff']

    print('=' * 80)
    print('   FuxiRec: A library for general recommendation.   ')
    print('=' * 80)
    
    print('Classic Baseline Models:')
    print('   '.join(classic_baselines))
    print('-' * 100)
    print('Graph-Based Baseline Models:')
    print('   '.join(graph_baselines))
    print('-' * 100)
    print('Graph Signal Processing Models:')
    print('   '.join(graph_signal))
    print('-' * 100)
    print('Hypergraph Models:')
    print('   '.join(hypergraph))
    print('-' * 100)
    print('Social Recommendations Models:')
    print('   '.join(social_recommendations))
    print('-' * 100)
    print('Negative Sampling Models:')
    print('   '.join(negative_sampling))
    print('-' * 100)
    print('Self-Supervised Graph-Based Models:')
    for i in range(len(ssl_graph_models)//7 + 1):   
        print('   '.join(ssl_graph_models[i*7:(i+1)*7]))
    print('=' * 80)
    print('Sequential Baseline Models:')
    print('   '.join(sequential_baselines))
    print('-' * 100)
    print('Self-Supervised Sequential Models:')
    print('   '.join(ssl_sequential_models))
    print('-' * 100)
    print('Autoencoder-Based Models:')
    print('   '.join(autoencoder_baselines))
    print('=' * 100)
    print('Diffusion Models:')
    print('   '.join(diffusion_models))
    print('=' * 100)
    print('Test template:')
    print('   '.join(test_models))
    print('-' * 80)
    model = input('Please enter the model you want to run:').lower()
    import time

    s = time.time()
    code2model = {'c1':'MF', 'c2':'NeuMF',
                  'g1':'APPNP', 'g2':'DirectAU', 'g3':'LightGCN', 'g4':'LTGNN', 'g5':'ForwardRec', 'g6':'HGCF', 'g7':'LinkProp',
                  'gs1':'LGCN', 'gs2':'PGSP', 'gs3':'JGCF', 'gs4': 'SGFCF', 'gs5': 'ChebyCF',
                  'hg1':'DHCF', 'hg2':'HCCF',
                  'sr1':'DiffNet', 'sr2':'DiffNetPlus', 'sr3':'MHCN', 'sr4':'SEPT',
                  'ns1':'MixGCF', 'ns2':'DENS', 'ns3':'DINS', 'ns4':'AHNS',
                  'sg1':'SGL', 'sg2':'BUIR', 'sg3':'SSL4Rec', 'sg4':'SimGCL', 'sg5':'NCL', 'sg6':'AdaGCL', 'sg7':'SelfCF', \
                      'sg8':'LightGCL', 'sg9':'XSimGCL', 'sg10':'EGCF', 'sg11':'SCCF', 'sg12':'RecDCL', 'sg13': 'SGCL',
                  's1':'SASRec', 's2':'FBABRF',
                  'ss1':'CL4SRec', 'ss2':'DuoRec', 'ss3':'BERT4Rec',
                  'a1':'CDAE', 'a2':'MultiVAE',
                  'd1': 'CODIGEM', 'd2': 'DiffRec', 'd3': 'L_DiffRec', 'd4': 'BSPM', 'd5': 'GiffCF', 'd6': 'DDRM', 'd7': 'PreferDiff', 'd8': 'HDRM', 'd9':'TV_Diff',
                  '1': 'test', '2': 'CoRec', '3': 'DiffGraph','4':'GumbelDiff'}
    try:
        conf = ModelConf('./conf/' + code2model[model] + '.conf')
    except:
        print('Wrong model name!')
        exit(-1)
    rec = FuxiRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
