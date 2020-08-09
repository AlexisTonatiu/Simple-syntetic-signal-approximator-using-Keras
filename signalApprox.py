import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

'''
Function to genereta the "data set". Pretty simple thing in this case.
Datapoints is going to set how large the input vector is going to be
'''
def genDataTrain(datapoints):
    base = np.linspace(0, 2 *np.pi, datapoints) 
    
    train_y = np.array([(np.sin(base) + 1) / 2, (sig.sawtooth(base) + 1) / 2, (sig.sawtooth(base, 1 / 2) + 1) / 2, (sig.square(base) + 1) / 2])
    
    train_x = base - base.min()
    train_x = train_x / train_x.max()
    
    return train_x.reshape(-1,1), np.transpose(train_y)

'''
Function where the model is created 
'''
def modeloSenales(input_tr):
    modelo = tf.keras.Sequential(name='Modelo_videoclase')
    modelo.add(Dense(128,input_shape=[1], activation='relu'))
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dense(32, activation='relu'))
    modelo.add(Dense(4, activation='relu')) 

    modelo.compile(optimizer='adam', loss='mse', metrics=['mse'])

    return modelo

'''
Function to generate periodic signal from the model approximations
You can choose whether you plot it or you save the vector into a variable 
'''
def generadorNet(frecuencia = 1, dataPointsPerCiclo = 8, naturaleza = 'seno', graficar = False):
    t = np.linspace(0, 2 * np.pi, dataPointsPerCiclo)
    t = np.subtract(t, t.min())
    t = np.divide(t, t.max())

    signal = gen_seno(t)

    for f in range(1, frecuencia):
        signal = np.concatenate([signal, gen_seno(t)])

    if graficar and naturaleza == 'todas':
        for i in range(4):
            plt.figure(figsize=(20,8))
            plt.subplot(4, 1, i + 1)
            plt.plot(signal[:,i])
        return 

    if naturaleza == 'seno':
        signal_out = signal[:, 0]
        if graficar:
            plt.figure(figsize=(20,5))
            plt.plot(signal_out)
            return

    if naturaleza == 'sierra':
        signal_out = signal[:, 1]
        if graficar:
            plt.figure(figsize=(20,5))
            plt.plot(signal_out)
            return

    if naturaleza == 'triangular':
        signal_out = signal[:, 2]
        if graficar:
            plt.figure(figsize=(20,5))
            plt.plot(signal_out)
            return

    if naturaleza == 'cuadrada':
        signal_out = signal[:, 3]
        if graficar:
            plt.figure(figsize=(20,5))
            plt.plot(signal_out)
            return

    if naturaleza == 'todas':
        return signal[:, 0], signal[:, 1], signal[:, 2], signal[:, 3]

    return signal_out


inp, out = genDataTrain(40000) # Creating the actual data to train the model
gen_seno = modeloSenales(inp) 

# Training - Uncomment to train
#history = gen_seno.fit(inp, out, epochs=100 , batch_size=64, verbose=2) 

# Testing
generadorNet(8, 4000, 'todas', True)