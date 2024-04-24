import random
import numpy
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
#from thundersvm import SVC

from sklearn import svm
# Librerias de ayuda
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import warnings
#with tf.device('/cpu:0'):

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
#toolbox.register("indices", create_individual) # aquí debemos registar una función que generar una muestra de individuo
#print(toolbox.indices())
# Generación de inviduos y población

toolbox.register("attr_int", random.randint, 0, 1)
toolbox.register("attr_int2", random.randint, -5, 15)
toolbox.register("attr_int3", random.randint, -15, 3)
toolbox.register("attr_int4", random.randint, 3, 7)
toolbox.register("individual", tools.initCycle, creator.Individual,
             (toolbox.attr_int, toolbox.attr_int2,toolbox.attr_int3, toolbox.attr_int4),
             n=1)
#toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
POP_SIZE=200
toolbox.register("population", tools.initRepeat, list, toolbox.individual, POP_SIZE) #
ind = toolbox. individual() # creamos un individuo aleatorio
print(ind.fitness.values) # en fitness.values se guardará el fitness
pop = toolbox.population() # creamos una población aleatoria
print(pop[:5]) # imprimimos los 5 primeros individuos
toolbox.register("mate", tools.cxOnePoint)                       
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=(1/4)) 
toolbox.register("select", tools.selTournament, tournsize=6)  

    
def evalua_accuracy(individual):
    
    
    if individual[0] == 0:
        kernel1 = "rbf"
    else:
        kernel1 = "poly"
    warnings.simplefilter(action='ignore', category=FutureWarning) 
    data = pd.read_csv('taiwan.csv.csv')
    target = data['Bankrupt?']
    del data['Bankrupt?']
    dato = data[:1]
    SMOTE_oversample = SMOTE()
    data,target = SMOTE_oversample.fit_resample(data,target)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=42)
    std_slc =  StandardScaler()
    std_slc.fit(X_train)
    X_train_std = std_slc.transform(X_train)
    X_test_std = std_slc.transform(X_test)
    pca = PCA(n_components=93)# adjust yourself
    pca.fit(X_train_std)
    X_t_train_std = pca.transform(X_train_std)
    X_t_test_std = pca.transform(X_test_std)
    parameterc=2**individual[1]
    pgamma = 2**individual[2]
    print("Parametro C ",parameterc," parametro gamma ",pgamma, " kernell ",kernel1, " degree", individual[3])
    if int(individual[3]) < 3 or int(individual[3]) > 7:
        accuracy=0.0
        return (accuracy),
    elif parameterc == 1:
        accuracy=0.0
        return (accuracy),
    else:
        clf = svm.SVC(kernel=kernel1, C=parameterc, gamma=pgamma, degree=individual[3])
        clf.fit(X_t_train_std, y_train)
        
        y_pred = clf.predict(X_t_test_std)
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)
        return (accuracy),
toolbox.register("evaluate", evalua_accuracy)

def plot_evolucion(log):
    gen = log.select("gen")
    fit_mins = log.select("min")
    fit_maxs = log.select("max")
    fit_ave = log.select("avg")

    fig, ax1 = plt.subplots()
    ax1.plot(gen, fit_mins, "b")
    ax1.plot(gen, fit_maxs, "r")
    ax1.plot(gen, fit_ave, "--k")
    ax1.fill_between(gen, fit_mins, fit_maxs, 
                     where=fit_maxs >= fit_mins, 
                     facecolor="g", alpha=0.2)
    ax1.set_xlabel("Generación")
    ax1.set_ylabel("Fitness")
    ax1.legend(["Min", "Max", "Avg"])
    ax1.set_ylim([0, 1])
    plt.grid(True)
    plt.savefig("EvolucionTSP.eps", dpi=300)
def main():
    random.seed(42) # ajuste de la semilla del generador de números aleatorios
    CXPB, MUTPB, NGEN = 0.7,(1/4), 30
    pop = toolbox.population() # creamos la población inicial 
    hof = tools.HallOfFame(1) 
    stats = tools.Statistics(lambda ind: ind.fitness.values) 
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    logbook = tools.Logbook()     
    pop, logbook = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=stats, halloffame=hof, verbose=True)
    return hof, logbook
best, log = main()
print("Mejor fitness: %f" %best[0].fitness.values)
print("Mejor individuo %s" %best[0])
plot_evolucion(log) # mostamos la evolución