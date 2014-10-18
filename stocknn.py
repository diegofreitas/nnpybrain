__author__ = 'diego.freitas'


from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer



class NeuralNetwork:

    n = RecurrentNetwork()

    def __init__(self):
        inLayer = LinearLayer(1)
        hiddenLayer = SigmoidLayer(10)
        hiddenLayer2 = SigmoidLayer(5)
        outLayer = LinearLayer(1)

        self.n.addInputModule(inLayer)
        self.n.addModule(hiddenLayer)
        self.n.addModule(hiddenLayer2)
        self.n.addOutputModule(outLayer)

        in_to_hidden = FullConnection(inLayer, hiddenLayer)
        hidden_to_hidden2 = FullConnection(hiddenLayer, hiddenLayer2)
        hidden2_to_out = FullConnection(hiddenLayer, outLayer)

        self.n.addConnection(in_to_hidden)
        self.n.addConnection(hidden_to_hidden2)
        self.n.addConnection(hidden2_to_out)

        self.n.addRecurrentConnection(FullConnection(hiddenLayer, hiddenLayer, name='recurrent'))
        self.n.addRecurrentConnection(FullConnection(hiddenLayer2, hiddenLayer2, name='recurrent2'))

        self.n.sortModules()
    #print(n)

    def train(self, dataset):
        TrainDS, TestDS = dataset.splitWithProportion(0.8)
        trainer = BackpropTrainer(self.n, TrainDS);
        for i in xrange(6):
            trainer.trainEpochs(1)
        trainer.testOnData(TestDS, True)

    def predict(self,features):
        return self.n.activate(features)



    #pickle.dump(n, open('testNetwork.dump', 'w'))
    #n = pickle.load(open('testNetwork.dump'))

    #n.sorted = False
    #n.sortModules()


