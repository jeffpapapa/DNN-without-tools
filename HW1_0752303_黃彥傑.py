
# coding: utf-8

# In[1]:


import numpy as np #basic tutorial
import random #shuffle datas
import pandas as pd #load files
import math #computing log
import matplotlib.pyplot as plt #plot results


# In[2]:


class DNN(object):
    #Start building DNN
    #We need conponents of 
    #0. Initail information   //done
    #1. fordward propagation  //done
    #2. backward propagation  //done
    #3. SGD process           //done
    #4. Accurancy             //done  
    #5. activation functions  //done
    #6. objective function    //done
    #7. Leraning rate         //done
    #8. Records(Entro, train, test)  //done
    #9. Plot results          //done
    #10.Debugsssssssssssssssss//????
    #11. softmax?
    
    def __init__(self, layer_size):
        self.layer_num = len(layer_size)
        self.layer_size = layer_size
        self.entropy_record = []
        self.training_error_record = []
        self.testing_error_record = []
        
        
        ## Now, build initial weight and bias
        ## w'*layer + b = next layer
        ## (next size * now size) * (now size * 1) + (next size) = (next size)
        ## bias = next size
        self.weight = [np.random.randn(now_size, next_size) for (now_size, next_size) in zip(layer_size[1:], layer_size[:-1])]
        #self.weight = [np.random.uniform(-(np.sqrt(6/(now_size+next_size))),(np.sqrt(6/(now_size+next_size))),(now_size, next_size)) for (now_size, next_size) in zip(layer_size[1:], layer_size[:-1])]
        self.bias = [np.random.randn(next_size) for next_size in layer_size[1:]] 
        self.velocity_w =  [np.zeros(w.shape) for w in self.weight]
        self.velocity_b = [np.zeros(b.shape) for b in self.bias]
    
    def forward(self, x):
        #last layer should use sigmoid
        #others use relu
        for w, b in zip(self.weight[:-1], self.bias[:-1]):
            ##########
            #x = relu(np.dot(w, x) + b)
            x = sigmoid(np.dot(w, x) + b)
        y = softmax(sigmoid(np.dot(self.weight[-1], x) + self.bias[-1]))
        
        return y
    
    def backpropagation(self, x, y):
        #For x, it's only one data
        #First, feed forward and store every value
        #print('x : ', x)
        n_val = x
        neural_value = [x]
        linear_value = []
        
        #last layer should use sigmoid, while others use relu
        for w, b in zip(self.weight[:-1],self.bias[:-1]):
            z = np.dot(w,n_val)+b
            linear_value.append(z)
            #print(z) ####################
            #n_val = relu(z)
            n_val = sigmoid(z)
            neural_value.append(n_val)
        
        z = np.dot(self.weight[-1],n_val)+self.bias[-1]
        linear_value.append(z)
        n_val = softmax(sigmoid(z))
        neural_value.append(n_val)
            
        #Now, we calculate delta values
        #we need : 1. last layer gap, 2. diff. of activation, 3. previous neural value
        #times them together
        #However, we should handle last layer first
        delta_w = [np.zeros(w.shape) for w in self.weight]
        delta_b = [np.zeros(b.shape) for b in self.bias]
        
        Error = objective_function(neural_value[-1],y)
        #print((neural_value[-1]-y))
        #print(neural_value[-1])
        #print(y)
        #print(objective_function(neural_value[-1],y))
        
        delta = d_obj(neural_value[-1],y)
        #delta = ((neural_value[-1] - y))*(d_sigmoid(linear_value[-1]))
        #delta = d_obj(neural_value[-1],y)*Error*(d_sigmoid(linear_value[-1]))
        #delta = d_obj(neural_value[-1],y)*(d_sigmoid(linear_value[-1]))
        
#         print(delta)
#         #we should times softmax layer first
#         delta = np.dot(delta.reshape(len(delta),1),delta.reshape(len(delta),1).transpose())
#         for i in range(0,len(delta)):
#             for j in range(0,len(delta)):
#                 if(i==j):
#                     delta[i][j] = delta[i][j]*(1-delta[i][j])
#                 else:
#                     delta[i][j] = (-1)*delta[i][j]
        
        
#         print(delta)
        #for bias, only 1 dimension
        delta_b[-1] = delta
        #for weight, should be previous layer's dimension
        #and every neural times delta

        delta_w[-1] = np.dot(delta.reshape(len(delta),1), np.array(neural_value[-2]).reshape(len(np.array(neural_value[-2])),1).transpose())
        #Till now, last layer has done computing delta value
        #Next, we should compute rest layers' delta value
        #by timing their 'next layer' (chain rule)
        #Note that these activation are using relu function
        
        #second layer(first layer is input) to (n-1) layer
        for layer_index in range(2,(self.layer_num)):
            #print(self.weight[-layer_index-2])
            # times w, then times derivative of relu
            #print(layer_index)
            #print(self.weight[-layer_index].shape) #########################
            #delta = np.dot(self.weight[-layer_index+1].transpose(), delta)*d_relu(linear_value[-layer_index])
            delta = np.dot(self.weight[-layer_index+1].transpose(), delta)*d_sigmoid(linear_value[-layer_index])
            delta_b[-layer_index] = delta
            #delta_w need to times previous neural value
            #print(layer_index)
            #print(delta)

            #print(self.weight)
            #print(neural_value)
            #print(self.layer_size)
            #print(delta_w[-layer_index-1])
            delta_w[-layer_index] = np.dot(delta.reshape(len(delta),1), np.array(neural_value[-layer_index-1]).reshape(len(np.array(neural_value[-layer_index-1])),1).transpose())
            #np.dot(delta, np.array(neural_value[-layer_index-2]).transpose())
        
        #Done, return delta table
        return delta_b, delta_w, (np.sum(Error))
    
    #Now, update parameters for every minibatch
    #for every mini-batch, we need to sum up their back prop value
    #then update weight and bias per mini-batch
    def update_parameters(self, mini_batch, learning_rate, training_data, testing_data, alpha):
        
        
        delta_w_total = [np.zeros(w.shape) for w in self.weight]
        delta_b_total = [np.zeros(b.shape) for b in self.bias]
        
        total_entropy = 0
        for x,y in mini_batch:
            delta_b, delta_w, entropy = self.backpropagation(x, y)
            delta_w_total = [dwt+dw for dwt,dw in zip(delta_w_total,delta_w)]
            delta_b_total = [dbt+db for dbt,db in zip(delta_b_total,delta_b)]
            total_entropy += entropy
        
#         w_norm = 0
#         for layer in delta_w_total:
#             for neural in layer:
#                 for num in neural:
#                     #print(num)
#                     w_norm+=num*num
#         w_norm = np.sqrt(w_norm)

#         b_norm = 0
#         for layer in delta_b_total:
#             for num in layer:
#                 b_norm+=num*num
#         b_norm = np.sqrt(b_norm)
 
#         step_size_w = learning_rate*w_norm/(1-alpha)
#         step_size_b = learning_rate*b_norm/(1-alpha)
        step_size_w = learning_rate
        step_size_b = learning_rate
        
        self.velocity_w = [alpha*v - step_size_w*(dwt/len(mini_batch)) for v,dwt in zip(self.velocity_w, delta_w_total)]
        self.velocity_b = [alpha*v - step_size_b*(dbt/len(mini_batch)) for v,dbt in zip(self.velocity_b, delta_b_total)]
        self.weight = [w + v for w,v in zip(self.weight, self.velocity_w)]
        self.bias = [b + v for b,v in zip(self.bias, self.velocity_b)]
        
        #Now, renew weight and bias
        #print(delta_w_total)
#         self.weight = [w - learning_rate*(dwt/len(mini_batch)) for w,dwt in zip(self.weight, delta_w_total)]
#         self.bias = [b - learning_rate*(dbt/len(mini_batch)) for b,dbt in zip(self.bias, delta_b_total)]
        
        self.entropy_record.append(total_entropy/len(mini_batch))
        self.training_error_record.append(self.accuracy(training_data)/len(training_data))
        self.testing_error_record.append(self.accuracy(testing_data)/len(testing_data))
        #print(len(self.training_error_record))
    
    def accuracy(self, data):
        #Last layer are equal 2 in this question
        #that is, surfive/dead are separated into 2 neural
        #the answer will be the largest neural, which we can use argmax
        answers = [(np.argmax(self.forward(x)), np.argmax(y)) for (x,y) in data]
        
        correct_num = 0
        for x,y in answers:
            #print(x)
            #print(y)
            if(x==y):
                correct_num+=1
        
        return correct_num
    
    def SGD(self, training_data, mini_batch_size, epoch, start_learning_rate, testing_data, alpha):
        learning_rate = start_learning_rate
        for epoch_num in range(0,epoch):
            #shuffle data
            random.shuffle(training_data)
            
            #split into mini batch
            mini_batches = [training_data[i:i+mini_batch_size] for i in range(0,len(training_data), mini_batch_size)]
            
            
            for mini_batch in mini_batches:
                try:
                    if(self.training_error_record[-1]<=self.training_error_record[-2]):
                        #then it's increasing, we should slower(decrease) learning rate
                        learning_rate = learning_rate*0.999
                        if(learning_rate<=0.0001):
                            #lower bound of learning rate
                            learning_rate = 0.0001
                        self.update_parameters(mini_batch, learning_rate, training_data, testing_data)
                    else:
                        #learning_rate = learning_rate*0.99
                        self.update_parameters(mini_batch, learning_rate, training_data, testing_data)
                except:
                    #print(mini_batch)
                    self.update_parameters(mini_batch, learning_rate, training_data, testing_data, alpha)
            
            print('{0} epoch / training accuracy : {1} / testing accuracy : {2} / lr : {3}'.format((epoch_num+1), self.training_error_record[-1], self.testing_error_record[-1],learning_rate))
        
    def plot_results(self, mini_batch_size):
        #Entropy
        
        y = [self.entropy_record[i] for i in range(0,len(self.entropy_record), int(800/mini_batch_size))]
        x = list(range(0,len(y)))
        #y = self.entropy_record
        #x = list(range(0,len(y)))
        #print('ya')
        plt.plot(x,y)
        #plt.plot(list(range(0,len(self.entropy_record))), self.entropy_record)
        plt.xlabel('Numbers of epoch')
        #plt.xlabel('Numbers of iteration')
        plt.ylabel('Average of cross entropy')
        plt.title('training loss')
        plt.show()
        
        y = [1-self.training_error_record[i] for i in range(0,len(self.training_error_record), int(800/mini_batch_size))]
        x = list(range(0,len(y)))
        #training_error_rate = [1-acc for acc in self.training_error_record]
        #plt.plot(list(range(0,len(training_error_rate))), training_error_rate)
        plt.plot(x,y)
        plt.xlabel('Numbers of epoch')
        #plt.xlabel('Numbers of iteration')
        plt.ylabel('Error rate')
        plt.title('training error rate')
        plt.show()
        
        
        y = [1-self.testing_error_record[i] for i in range(0,len(self.testing_error_record), int(800/mini_batch_size))]
        x = list(range(0,len(y)))
        plt.plot(x,y)
        #testing_error_rate = [1-acc for acc in self.testing_error_record]
        #plt.plot(list(range(0,len(testing_error_rate))), testing_error_rate)
        plt.xlabel('Numbers of epoch')
        #plt.xlabel('Numbers of iteration')
        plt.ylabel('Error rate')
        plt.title('testing error rate')
        plt.show()
def relu(x):
    try:
        z = []
        #print(x)
        for num in x:
            z.append(max(0,num))
        return z
    except:
        return(max(0,x))

def d_relu(x):    
    #print(x)
    try:
        z = []
        for num in x:
            if(num>=0):
                z.append(float(1.0))
            else:
                z.append(float(0.0))
        return z
    except:
        if(x>=0):
            return (float(1.0))
        else:
            return (float(0.0))

def sigmoid(x):
    return (1.0/(1.0 + np.exp(-x)))

def d_sigmoid(x):
    y = sigmoid(x)
    return y*(1 - y) 

def softmax(x):
    return x/(sum(x))

def objective_function(pre_y, true_y):
#def objective_function(a, y):
    Error = []
    
    for a,b in zip((pre_y), (true_y)):
        #print(a,b)
        if(a==0):
            a = 10**(-10)
        Error.append(b*(math.log(a)))
        #Error.append(b*(math.log(a))+(1-b)*(math.log(1-a)))
        #print('Error : ',Error)
    #print(Error)
    #Error = [b*(math.log(a)) for a,b in zip(pre_y, true_y)]
    return (np.dot((-1),(np.array(Error))))
    #return (np.nan_to_num(-true_y*np.log(pre_y)-(1-true_y)*np.log(1-pre_y)))

def d_obj(pre_y, true_y):
    Error = []
    count = 0
    for a,b in zip((pre_y), (true_y)):
        #print(a,b)
        Error.append(a-b)
        if(a==0):
            a = 0.00000001
        elif(a==1):
            a = 0.99999999
        #Error.append((-b/a))
        #Error.append((a-b)/(a*(1-a)))
        #print('Error : ',Error)
    #print(Error)
    #Error = [b*(math.log(a)) for a,b in zip(pre_y, true_y)]
    #return np.dot((-1),((Error)))
    return np.array(Error)


# # p1 : Desine own structures

# In[3]:


###Now, start dealing with data preprocessing

data = pd.read_csv('titanic.csv')
answers = np.array(pd.get_dummies(data['Survived'].values).values)
data = data.drop(columns=['Survived'])
for num in range(0,len(data)):
    data.loc[num,'Age'] = data.loc[num,'Age']/(np.max(data['Age'])-np.min(data['Age']))
    data.loc[num,'Fare'] = data.loc[num,'Fare']/(np.max(data['Fare'])-np.min(data['Fare']))
# Pclass = pd.get_dummies(data['Pclass'])
# data = data.drop(columns=['Pclass'])
# data = data.join(Pclass)
    
training_data = data[:800]
testing_data = data[800:]
training_ans = answers[0:800]
testing_ans = answers[800:]


train = [[a,b] for a, b in zip(np.array(training_data.values), training_ans)]
test = [[a,b] for a, b in zip(np.array(testing_data.values), testing_ans)]


# In[4]:



My_network = DNN([6,32,32,32,2])
mini_batch_size = 50
epoch = 100
start_learning_rate = 0.02
alpha = 0.99
My_network.SGD(train, mini_batch_size, epoch, start_learning_rate, test, alpha)


# In[5]:


My_network.plot_results(mini_batch_size)


# # p2 : Structures with [6,3,3,2]

# In[59]:


###Now, start dealing with data preprocessing

data = pd.read_csv('titanic.csv')
answers = np.array(pd.get_dummies(data['Survived'].values).values)
data = data.drop(columns=['Survived'])
for num in range(0,len(data)):
    data.loc[num,'Age'] = data.loc[num,'Age']/(np.max(data['Age'])-np.min(data['Age']))
    data.loc[num,'Fare'] = data.loc[num,'Fare']/(np.max(data['Fare'])-np.min(data['Fare']))
#Pclass = pd.get_dummies(data['Pclass'])
#data = data.drop(columns=['Pclass'])
#data = data.join(Pclass)
    
training_data = data[:800]
testing_data = data[800:]
training_ans = answers[0:800]
testing_ans = answers[800:]


train = [[a,b] for a, b in zip(np.array(training_data.values), training_ans)]
test = [[a,b] for a, b in zip(np.array(testing_data.values), testing_ans)]

First_network = DNN([6,3,3,2])
mini_batch_size = 50
epoch = 100
start_learning_rate = 0.015
alpha = 0.99
First_network.SGD(train, mini_batch_size, epoch, start_learning_rate, test, alpha)


# In[60]:


First_network.plot_results(mini_batch_size)


# # p3: normalization?

# In[8]:


###Now, start dealing with data preprocessing

data = pd.read_csv('titanic.csv')
answers = np.array(pd.get_dummies(data['Survived'].values).values)
data = data.drop(columns=['Survived'])
# for num in range(0,len(data)):
#     data.loc[num,'Age'] = data.loc[num,'Age']/(np.max(data['Age'])-np.min(data['Age']))
#     data.loc[num,'Fare'] = data.loc[num,'Fare']/(np.max(data['Fare'])-np.min(data['Fare']))
# Pclass = pd.get_dummies(data['Pclass'])
# data = data.drop(columns=['Pclass'])
# data = data.join(Pclass)
    
training_data = data[:800]
testing_data = data[800:]
training_ans = answers[0:800]
testing_ans = answers[800:]


train = [[a,b] for a, b in zip(np.array(training_data.values), training_ans)]
test = [[a,b] for a, b in zip(np.array(testing_data.values), testing_ans)]

sec_network = DNN([6,32,32,32,2])
mini_batch_size = 50
epoch = 100
start_learning_rate = 0.02
alpha = 0.99
sec_network.SGD(train, mini_batch_size, epoch, start_learning_rate, test, alpha)


# In[9]:


sec_network.plot_results(mini_batch_size)


# In[10]:


###Now, start dealing with data preprocessing

data = pd.read_csv('titanic.csv')
answers = np.array(pd.get_dummies(data['Survived'].values).values)
data = data.drop(columns=['Survived'])
for num in range(0,len(data)):
#     data.loc[num,'Age'] = data.loc[num,'Age']/(np.max(data['Age'])-np.min(data['Age']))
     data.loc[num,'Fare'] = data.loc[num,'Fare']/(np.max(data['Fare'])-np.min(data['Fare']))
# Pclass = pd.get_dummies(data['Pclass'])
# data = data.drop(columns=['Pclass'])
# data = data.join(Pclass)
    
training_data = data[:800]
testing_data = data[800:]
training_ans = answers[0:800]
testing_ans = answers[800:]


train = [[a,b] for a, b in zip(np.array(training_data.values), training_ans)]
test = [[a,b] for a, b in zip(np.array(testing_data.values), testing_ans)]

sec_network = DNN([6,32,32,32,2])
mini_batch_size = 50
epoch = 100
start_learning_rate = 0.02
alpha = 0.99
sec_network.SGD(train, mini_batch_size, epoch, start_learning_rate, test, alpha)


# In[11]:


sec_network.plot_results(mini_batch_size)


# # p4 : What feature affects most?

# ## However, from p3 result, 'Age' does not affect most for sure

# In[12]:


data = pd.read_csv('titanic.csv')
data.corr()


# 'Sex' > 'Pclass' > 'Fare' > 'Parch' > 'SibSp' > 'Age'

# ## p5 : Should we use one-hot for Pclass?

# In[57]:


###Now, start dealing with data preprocessing

data = pd.read_csv('titanic.csv')
answers = np.array(pd.get_dummies(data['Survived'].values).values)
data = data.drop(columns=['Survived'])
for num in range(0,len(data)):
    data.loc[num,'Age'] = data.loc[num,'Age']/(np.max(data['Age'])-np.min(data['Age']))
    data.loc[num,'Fare'] = data.loc[num,'Fare']/(np.max(data['Fare'])-np.min(data['Fare']))
Pclass = pd.get_dummies(data['Pclass'])
data = data.drop(columns=['Pclass'])
data = data.join(Pclass)
    
training_data = data[:800]
testing_data = data[800:]
training_ans = answers[0:800]
testing_ans = answers[800:]


train = [[a,b] for a, b in zip(np.array(training_data.values), training_ans)]
test = [[a,b] for a, b in zip(np.array(testing_data.values), testing_ans)]

First_network = DNN([8,32,32,32,2])
mini_batch_size = 50
epoch = 100
start_learning_rate = 0.02
alpha = 0.99
First_network.SGD(train, mini_batch_size, epoch, start_learning_rate, test, alpha)


# In[58]:


First_network.plot_results(mini_batch_size)


# In[16]:


data = pd.read_csv('titanic.csv')
# answers = np.array(pd.get_dummies(data['Survived'].values).values)
# data = data.drop(columns=['Survived'])
# for num in range(0,len(data)):
#     data.loc[num,'Age'] = data.loc[num,'Age']/(np.max(data['Age'])-np.min(data['Age']))
#     data.loc[num,'Fare'] = data.loc[num,'Fare']/(np.max(data['Fare'])-np.min(data['Fare']))
Pclass = pd.get_dummies(data['Pclass'])
data = data.drop(columns=['Pclass'])
data = data.join(Pclass)
data.corr()

# survival : 1>2>3, fit as the distance without one-hot encoding

# ## p6 : Desine own data 

# In[56]:


### 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'
survive = [1,0,0.24,0,0,0.82]
print(survive)
print(np.argmax(My_network.forward(survive)))

print('\n')
dead = [3,1,0.75,1,2,0.14]
print(dead)
print(np.argmax(My_network.forward(dead)))

