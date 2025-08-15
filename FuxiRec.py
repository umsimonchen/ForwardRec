from data.loader import FileIO
import importlib

class FuxiRec(object):
    def __init__(self, config):
        self.social_data = []
        self.feature_data = []
        self.config = config
        self.training_data = FileIO.load_data_set(config['training.set'], config['model.type'])
        self.test_data = FileIO.load_data_set(config['test.set'], config['model.type'])

        self.kwargs = {}
        if config.contain('social.data'):
            social_data = FileIO.load_social_data(self.config['social.data'])
            self.kwargs['social.data'] = social_data
        # if config.contains('feature.data'):
        #     self.social_data = FileIO.loadFeature(config,self.config['feature.data'])
        print('Reading data and preprocessing...')

    def execute(self):
        # import the model module
        import_str = 'model.'+ self.config['model.type'] +'.' + self.config['model.name']
        module = importlib.import_module(import_str)
        RecommenderClass = getattr(module, self.config['model.name'])
        recommender = RecommenderClass(self.config,self.training_data,self.test_data,**self.kwargs)
        recommender.execute()
