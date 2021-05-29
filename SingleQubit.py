import pennylane as qml
import math
import wandb
from pennylane import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dev = qml.device('default.qubit', wires=2, shots=1000, analytic=False)
@qml.qnode(dev)
def qcircuit(param, data):
    
    embeddingCircuit(data)
    qml.Rot(param[0][0][0], param[0][0][1], param[0][0][2], wires = [0])
    qml.Rot(param[1][0][0], param[1][0][1], param[1][0][2], wires = [1])
    qml.CZ(wires=[1,0])
    
    embeddingCircuit(data)
    qml.Rot(param[0][1][0], param[0][1][1], param[0][1][2], wires = [0])
    qml.Rot(param[1][1][0], param[1][1][1], param[1][1][2], wires = [1])
    qml.CZ(wires=[1,0])
    # embeddingCircuit(data)
    
    # qml.Rot(param[0][1][0], param[0][1][1], param[0][1][2], wires = [0])
    # qml.Rot(param[1][1][0], param[1][1][1], param[1][1][2], wires = [1])
    # qml.CZ(wires=[1,0])
    #qml.Rot(param[1][0], param[1][1], param[1][2], wires = [0])
    # embeddingCircuit(data)
    # qml.Rot(param[2][0], param[2][1], param[2][2], wires = [0])
    return qml.expval(qml.PauliZ(0))


def embeddingCircuit(data):
    norm = np.linalg.norm(data)
    norm = norm if norm !=0 else 1
    for i in range(0, len(data)-1, 2):
        qml.RX(data[i]*2*math.pi/norm, wires=0)
        qml.RX(data[i+1]*2*math.pi/norm, wires=1)


def getPrediction(var, data=None):
    
    quantumOutput = qcircuit(var, data)
    # print("For", data)
    # print("Prediction: ", quantumOutput)
    return quantumOutput
    #no bias

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

def cost(params, X, Y):

    predictions = [getPrediction(params, x) for x in X]
    return lossFunction(Y, predictions)





dataSet = load_breast_cancer()
X = dataSet.data
Y = dataSet.target
Y = Y * 2 - np.ones(len(Y))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
# print(np.count_nonzero(Y_test==1))
# print(np.count_nonzero(Y_test==-1))
# np.random.seed()

numQubits = 2

numLayers = 2

varInit = 0.01 * np.random.randn(numQubits, numLayers, 3)
opt = qml.AdamOptimizer()
batchSize = 4
var = varInit
modelConfig = {
    "numQubits" : numQubits,
    "numLayers" : numLayers,
    "batchSize" : batchSize,
    "varInit" : varInit,
    "opt": "Default Adam"
}
wandb.init(project='qml-experiments', entity='pj747', config=modelConfig)
qcircuit(var, X_train[0])
circ = qcircuit.draw()
wandb.log({"circuit": str(circ)})
print("Sample Circuit:" , circ)
for it in range(300):
    batchIndex = np.random.randint(0, len(X_train), (batchSize,))
    X_batch = X_train[batchIndex]
    Y_batch = Y_train[batchIndex]
    var = opt.step(lambda v: cost(v, X_batch, Y_batch), var)
    print("Final Value: ", var)
    # Compute accuracy
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
            wandb.log({"train_cost":loss, "train_acc":a}, commit=False)
        wandb.log({"test_cost":test_loss, "test_acc":test_acc})
wandb.log({"parameters": var})
wandb.finish()








