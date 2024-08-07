from SELFRec import SELFRec
from util.conf import ModelConf

if __name__ == '__main__':
    # Register your model here
    graph_baselines = ['g1. LightGCN','g2. DirectAU','g3. MF','g4.ForwardRec', 'g5. LTGNN', 'g6. APPNP']
    graph_signal = ['gs1. LGCN', 'gs2. PGSP']
    hypergraph = ['hg1. DHCF', 'hg2. HCCF']
    social_recommendations = ['sr1. DiffNet', 'sr2. DiffNet++', 'sr3. MHCN', 'sr4. SEPT']
    negative_sampling = ['ns1. MixGCF', 'ns2. DENS']
    ssl_graph_models = ['sg1. SGL', 'sg2. SimGCL','sg3. BUIR', 'sg4. SelfCF', 'sg5. SSL4Rec', 'sg6. XSimGCL', 'sg7. NCL']
    sequential_baselines= ['s1. SASRec', 's2. FBABRF']
    ssl_sequential_models = ['ss1. CL4SRec','ss2. DuoRec','ss3. BERT4Rec']

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
    print('   '.join(ssl_graph_models))
    print('=' * 80)
    print('Sequential Baseline Models:')
    print('   '.join(sequential_baselines))
    print('-' * 100)
    print('Self-Supervised Sequential Models:')
    print('   '.join(ssl_sequential_models))
    print('=' * 80)
    model = input('Please enter the model you want to run:').lower()
    import time

    s = time.time()
    code2model = {'g1':'LightGCN', 'g2':'DirectAU', 'g3':'MF', 'g4':'ForwardRec', 'g5':'LTGNN', 'g6':'APPNP',
                  'gs1':'LGCN', 'gs2':'PGSP',
                  'hg1':'DHCF', 'hg2':'HCCF',
                  'sr1':'DiffNet', 'sr2':'DiffNetPlus', 'sr3':'MHCN', 'sr4':'SEPT',
                  'ns1':'MixGCF', 'ns2':'DENS',
                  'sg1':'SGL', 'sg2':'SimGCL', 'sg3':'BUIR', 'sg4':'SelfCF', 'sg5':'SSL4Rec', 'sg6':'XSimGCL','sg7':'NCL', 
                  's1':'SASRec', 's2':'FBABRF',
                  'ss1':'CL4SRec', 'ss2':'DuoRec', 'ss3':'BERT4Rec'}
    try:
        conf = ModelConf('./conf/' + code2model[model] + '.conf')
    except:
        print('Wrong model name!')
        exit(-1)
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
