import numpy as np
import os
#estructura = (30,20,20,3)

class GenericNeuralNetwork:
    def __init__(self,structure=0,number_of_agents=100):
        self.structure = structure
        self.number_of_agents = number_of_agents
    
    #se busca un modelo de red neuronal en una carpeta especificada
    #si existe carga el modelo y devuelve una variable con el modelo
    def load_agent(self,file_location):
        if os.path.exists(str(file_location + '/w1.csv')) == True:
            print('Modelo encontrado')
            print('Cargando modelo...')

            #variables para crear la direccion de cada parte del agente
            agent_part_counter = 1
            agent = []
            weight = '/' + 'w' + str(agent_part_counter) + '.csv'
            bias = '/' +'b' + str(agent_part_counter) + '.csv'

            #si existe esa parte del agente, se agrega al agente
            while os.path.exists(file_location + weight) == True:
                
                agent.append([np.array(np.loadtxt(file_location + weight,delimiter=',').tolist()),np.array([np.loadtxt(file_location + bias,delimiter=',').tolist()])])
                
                #se suma 1 para buscar la siguiente parte del agente
                agent_part_counter += 1
                
                weight = '/' + 'w' + str(agent_part_counter) + '.csv'
                bias = '/' +'b' + str(agent_part_counter) + '.csv'

        return agent
    
    def save_agent(self,file_location,agent):
        if os.path.exists(file_location) == True:
            #variables para crear la direccion de cada parte del agente
            agent_part_counter = 1

            for i in range(len(agent)):
                weight = '/' + 'w' + str(agent_part_counter) + '.csv'
                bias = '/' +'b' + str(agent_part_counter) + '.csv'

                np.savetxt(str(file_location+weight),agent[i][0],delimiter=',')
                np.savetxt(str(file_location+bias),agent[i][1],delimiter=',')

                #se suma 1 para guardar la siguiente parte del agente
                agent_part_counter += 1

    def create_agents(self):
        #Se crean los agentes siendo cada uno una lista de los pesos y biases de cada red neuronal
        agents = []
        for agent in range(self.number_of_agents):
            agent = []
            for layer in range(len(self.structure)-1):
                #Primero se crean los pesos y luego sus biases
                agent.append([2*np.random.random_sample((self.structure[layer],self.structure[layer+1]))-1,2*np.random.random_sample((1,self.structure[layer+1]))-1])
            agents.append(agent)

        #Se devuelve una lista con todos los agentes generados
        return agents

    def crossover(self,A1,A2,mutation_rate=5):
        #---------------------------------------
        #CROSSOVER Y MUTACION DE PESOS
        #'A1' y 'A2' significa Agente 1 y Agente 2
        #Mutation rate es la probabilidad de mutacion, va desde 0 hasta 100
        agent1 = []
        agent2 = []
        for i in A1:
            agent1.append(list([np.array(i[0]),np.array(i[1])]))
        for i in A2:
            agent2.append(list([np.array(i[0]),np.array(i[1])]))

        #Se crea un punto aleatorio para iniciar el crossover

        all_weights_len = 0
        for i in agent1:
            all_weights_len += len(i[0])

        rand_point_weight = np.random.randint(1,all_weights_len)
        rand_point_counter = 0
        #Se realiza el crossover de los pesos de ambos padres
        for layer in range(len(agent1)):
            for fila in range(len(agent1[layer][0])):
                if rand_point_counter <= rand_point_weight:
                    rand_point_counter += 1
                    for columna in range(len(agent1[layer][0][0])):
                        #Se intercambian los pesos de cada numpy array
                        agent1[layer][0][fila][columna], agent2[layer][0][fila][columna] = agent2[layer][0][fila][columna], agent1[layer][0][fila][columna]
                        
        #Se aplica la mutacion dependiento del mutation rate, que va desde 0 hasta 100
        if np.random.randint(1,100) <= mutation_rate:
            rand_layer = np.random.randint(len(agent1))
            rand_fila = np.random.randint(len(agent1[rand_layer][0]))
            rand_columna = np.random.randint(len(agent1[rand_layer][0][0]))

            agent1[rand_layer][0][rand_fila][rand_columna] = 2*np.random.random_sample()-1
            agent2[rand_layer][0][rand_fila][rand_columna] = 2*np.random.random_sample()-1

        #Se crea un punto aleatorio para iniciar el crossover
        all_bias_len = 0
        for i in agent1:
            all_bias_len += len(i[1])

        rand_point_bias = np.random.randint(1,all_bias_len)
        rand_point_counter = 0
        #Se realiza el crossover de los biases de ambos padres
        for layer in range(len(agent1)):

            if rand_point_counter <= rand_point_bias:
                rand_point_counter += 1
                for fila in range(len(agent1[layer][1])):
                    for columna in range(len(agent1[layer][1][0])):
                        #Se intercambian los biases de cada numpy array
                        agent1[layer][1][fila][columna], agent2[layer][1][fila][columna] = agent2[layer][1][fila][columna], agent1[layer][1][fila][columna]

        #Se aplica la mutacion dependiento del mutation rate, que va desde 0 hasta 100
        if np.random.randint(1,100) <= mutation_rate:
            rand_layer = np.random.randint(len(agent1))
            rand_fila = np.random.randint(len(agent1[rand_layer][1]))
            rand_columna = np.random.randint(len(agent1[rand_layer][1][0]))
            
            agent1[rand_layer][1][rand_fila][rand_columna] = 2*np.random.random_sample()-1
            agent2[rand_layer][1][rand_fila][rand_columna] = 2*np.random.random_sample()-1

        return agent1, agent2

    def sigmoid(x):
        return 1 / (1 + np.exp(x))

    def feedforward_relu(self,x,agent):
        x = np.array(x)
        for i in range(len(agent)):
            bias = agent[i][1]
            z = np.maximum(0,((x.dot(agent[i][0]))+bias))
            x = z 
        return x
    
    def feedforward_sigmoid(self,x,agent):
        x = np.array(x)
        for i in range(len(agent)):
            bias = agent[i][1]
            z = GenericNeuralNetwork.sigmoid((x.dot(agent[i][0]))+bias)
            x = z
        return x
    
    def feedforward_sigmoid_relu(self,x,agent):
        x = np.array(x)
        for i in range(len(agent)-1):
            bias = agent[i][1]
            z = GenericNeuralNetwork.sigmoid((x.dot(agent[i][0]))+bias)
            x = z
        z = np.maximum(0,((x.dot(agent[-1][0]))+agent[-1][1]))
        x = z
        return x

    def feedforward_relu_sigmoid(self,x,agent):
        x = np.array(x)
        for i in range(len(agent)-1):
            bias = agent[i][1]
            z = np.maximum(0,((x.dot(agent[i][0]))+bias))
            x = z
        z = GenericNeuralNetwork.sigmoid(((x.dot(agent[-1][0]))+agent[-1][1]))
        x = z
        return x

'''
nn = GenericNeuralNetwork(((6,2,2,2)),2)
nn.create_agents()
agents = nn.create_agents()

agent1 = agents[0]
agent2 = agents[1]
print(nn.number_of_agents)
#print(agent1,'agent 1')
#print(agent2,'agent 2')

agent1, agent2 = nn.crossover(agent1,agent2,mutation_rate=90)
print(agent1,'agent 1')
#print(agent2,'agent 2')

x=[1,2,3,4,5,6]
forward = nn.feedforward_relu(x,agent1)
print(forward)
'''
    
