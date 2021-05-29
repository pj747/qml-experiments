import pennylane as qml
import math
import wandb
from pennylane import numpy as np
from types import SimpleNamespace
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataSet = load_breast_cancer()
X = dataSet.data
Y = dataSet.target
Y = Y * 2 - np.ones(len(Y))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)

def accuracy(labels, predictions):
    accuracy = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            accuracy = accuracy + 1
    accuracy = accuracy / len(labels)

    return accuracy

def lossFunction(labels, predictedLabels):
    loss = 0
    for l, p in zip(labels, predictedLabels):
        loss += (l-p) ** 2
    loss /= len(labels)
    return loss

def cost(qcircuit, params, X, Y):
    predictions = [getPrediction(qcircuit, params, x) for x in X]
    return lossFunction(Y, predictions)

def getPrediction(qcircuit, params, data=None):
    
    quantumOutput = qcircuit(params, data)
    # print("For", data)
    # print("Prediction: ", quantumOutput)
    return quantumOutput
    #no bias

def makeModel(config):
    numQubits = config.numQubits
    dev = qml.device(config.dev, wires=numQubits, shots=1000, analytic=False)
    @qml.qnode(dev)
    def qcircuit(param, data):
        for i in range(config.numLayers):
            embeddingCircuit(config, data)
            if config.fullEntangle:
                for j in range(numQubits):
                    for i in range(j):
                        qml.CZ(wires=[j,i])
            else:
                for j in range(numQubits-1):
                    qml.CZ(wires=[j,j+1])
            for j in range(numQubits):
                qml.Rot(param[j][i][0], param[j][i][1], param[j][i][2], wires = [j])
            
        return qml.expval(qml.PauliZ(0))

    def embeddingCircuit(config, data):
        norm = np.linalg.norm(data)
        norm = norm if norm !=0 else 1 
        for i in range(0, len(data)-numQubits, numQubits):
            for j in range(numQubits):
                qml.RX(data[i+j]*2*math.pi/norm, wires=j)
    
    return qcircuit


def createAndTrain(config, WandB = False):
    varInit = 0.01 * np.random.randn(config.numQubits, config.numLayers, 3)
    opt = qml.AdamOptimizer()
    batchSize = config.batchSize
    var = varInit
    qcircuit = makeModel(config)
    qcircuit(var, X_train[0])
    circ = qcircuit.draw()
    if WandB:
        wandb.log({"circuit": str(circ)})

    print("Sample Circuit:\n" , circ)
    numBatches = config.numBatches
    for it in range(numBatches):
        batchIndex = np.random.randint(0, len(X_train), (batchSize,))
        X_batch = X_train[batchIndex]
        Y_batch = Y_train[batchIndex]
        var = opt.step(lambda v: cost(qcircuit, v, X_batch, Y_batch), var)
        # Compute accuracy
        print("Computed batch ", it, var)
        predictions = []
        if(it%10 == 0):
            for x in X_test:
                #print(var, x)
                op = qcircuit(var, x)
                predictions.append(np.sign(op))

                #print(qcircuit.draw())
            acc = accuracy(Y_test, predictions)
            loss = lossFunction(Y_test, predictions)
            test_loss = float(loss)
            test_acc = float(acc)
            predictions = []
            if(it%50==0):
                for x in X_train:
                    op = qcircuit(var, x)
                    predictions.append(np.sign(op))
                acc = accuracy(Y_train, predictions)
                loss = lossFunction(Y_train, predictions)
                loss = float(loss)
                a = float(acc)
                if WandB:
                    wandb.log({"train_cost":loss, "train_acc":a}, commit=False)
            if WandB:
                wandb.log({"test_cost":test_loss, "test_acc":test_acc})
    if WandB:
        wandb.log({"parameters": var})
    wandb.finish()

def wandbRun(config):
    wandb.init(project='qml-experiments', entity='pj747', config=config)
    createAndTrain(config, WandB=True)

def wandbSweep():
    run = wandb.init()
    config = run.config
    createAndTrain(config, WandB=True)

if __name__ == "__main__":
    wandbSweep()
    # config = SimpleNamespace(
    #     numQubits = 2,
    #     numLayers = 1,
    #     dev = "default.qubit",
    #     batchSize = 4,
    #     numBatches = 300,
    #     fullEntangle = True
    # )
    # wandbRun(config)