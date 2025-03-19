from SELFRec import SELFRec
from util.conf import ModelConf

if __name__ == '__main__':
    # Register your model here
    graph_baselines = ['g1. MF','g2. APPNP','g3. DirectAU','g4. LightGCN', 'g5. LTGNN', 'g6. ForwardRec']
    graph_signal = ['gs1. LGCN', 'gs2. PGSP', 'gs3. JGCF', 'gs4. SGFCF']
    hypergraph = ['hg1. DHCF', 'hg2. HCCF']
    social_recommendations = ['sr1. DiffNet', 'sr2. DiffNet++', 'sr3. MHCN', 'sr4. SEPT']
    negative_sampling = ['ns1. MixGCF', 'ns2. DENS']
    ssl_graph_models = ['sg1. SGL', 'sg2. BUIR','sg3. SSL4Rec', 'sg4. SimGCL', 'sg5. NCL', 'sg6. AdaGCL', 'sg7.SelfCF ', \
                        'sg8. LightGCL', 'sg9. XSimGCL', 'sg10. EGCF', 'sg11. SCCF', 'sg12. RecDCL', 'sg13. SGCL']
    sequential_baselines= ['s1. SASRec', 's2. FBABRF']
    ssl_sequential_models = ['ss1. CL4SRec','ss2. DuoRec','ss3. BERT4Rec']
    diffusion_models = ['d1. DiffRec', 'd2. L-DiffRec', 'd3. BSPM', 'd4. GiffCF', 'd5. DDRM']
    test_models = ['1. test1', '2. CoRec', '3. DiffGraph']

    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    print('=' * 80)

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
    print('Diffusion-Based Models:')
    print('   '.join(diffusion_models))
    print('=' * 100)
    print('Test template:')
    print('   '.join(test_models))
    print('-' * 80)
    model = 'sg8'#input('Please enter the model you want to run:').lower()
    import time

    s = time.time()
    code2model = {'g1':'MF', 'g2':'APPNP', 'g3':'DirectAU', 'g4':'LightGCN', 'g5':'LTGNN', 'g6':'ForwardRec',
                  'gs1':'LGCN', 'gs2':'PGSP', 'gs3':'JGCF', 'gs4': 'SGFCF',
                  'hg1':'DHCF', 'hg2':'HCCF',
                  'sr1':'DiffNet', 'sr2':'DiffNetPlus', 'sr3':'MHCN', 'sr4':'SEPT',
                  'ns1':'MixGCF', 'ns2':'DENS',
                  'sg1':'SGL', 'sg2':'BUIR', 'sg3':'SSL4Rec', 'sg4':'SimGCL', 'sg5':'NCL', 'sg6':'AdaGCL', 'sg7':'SelfCF', \
                      'sg8':'LightGCL', 'sg9':'XSimGCL', 'sg10':'EGCF', 'sg11':'SCCF', 'sg12':'RecDCL', 'sg13': 'SGCL',
                  's1':'SASRec', 's2':'FBABRF',
                  'ss1':'CL4SRec', 'ss2':'DuoRec', 'ss3':'BERT4Rec',
                  'd1': 'DiffRec', 'd2': 'L_DiffRec', 'd3': 'BSPM', 'd4': 'GiffCF', 'd5': 'DDRM',
                  '1': 'test', '2': 'CoRec', '3': 'DiffGraph'}
    try:
        conf = ModelConf('./conf/' + code2model[model] + '.conf')
    except:
        print('Wrong model name!')
        exit(-1)
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
