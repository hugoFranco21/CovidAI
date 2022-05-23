import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import uuid

def get_corr_matrix(df, file_name="corr_mat"):
    f = open(file_name + ".txt", "w")
    f.write(df.corr().to_string())
    f.close

def save_model(m, model_id):
    f = open('history_models.txt', 'a')
    f.write('Model ' + model_id + '\n')
    f.write(str(m['w']) + '\n')
    f.write(str(m['b']) + '\n')
    f.write('learning rate = ' + str(m['learning_rate']) + '\n')
    f.write('num iterations = ' + str(m['num_iterations']) + '\n')
    f.write('train accuracy = ' + str(m['train_acc']) + '%\n')
    f.write('test_accuracy = ' + str(m['test_acc']) + '%\n')
    f.close()

def save_weight_bias(m):
    np.save('calc/weights', m['w'])
    f = open('calc/bias.txt', 'w')
    f.write(str(m['b']))
    f.close()

def load_dataset(split=0.2):
    df = pd.read_csv('dataset/covid-dataset.csv', header=0)
    df = df.sample(frac=1)
    df = df.drop(columns=['Wearing Masks','Sanitization from Market'])
    get_corr_matrix(df)
    aux = math.floor(df.shape[0] * split)
    num_train_rows = df.shape[0] - aux
    df_train = df.iloc[:num_train_rows,:]
    print("num_train_rows " + str(df_train.shape[0]))
    df_test = df.iloc[num_train_rows + 1:,:]
    print("num_test_rows " + str(df_test.shape[0]))
    train_set_x = df_train[["Breathing Problem","Fever","Dry Cough","Sore throat","Abroad travel",
        "Contact with COVID Patient","Attended Large Gathering", "Family working in Public Exposed Places"]]
    train_set_y = df_train[["COVID-19"]]
    test_set_x = df_test[["Breathing Problem","Fever","Dry Cough","Sore throat","Abroad travel",
        "Contact with COVID Patient","Attended Large Gathering", "Family working in Public Exposed Places"]]
    test_set_y = df_test[["COVID-19"]]
    
    #print(train_set_x.head())
    #print(train_set_y.head())
    #print(test_set_x.head())
    #print(test_set_y.head())
    return train_set_x.to_numpy(), \
        train_set_y.to_numpy(), \
        test_set_x.to_numpy(), \
        test_set_y.to_numpy(),

def k_fold_split(train_set_x, train_set_y, num_iterations, learning_rate, k_folds=5):
    weights, bias = initialize_with_zeros(train_set_x.shape[0])
    batch_size = math.floor(train_set_x.shape[1] / k_folds)
    for i in range(0,k_folds - 1):
        left_ind = batch_size*i
        if i > 0:
            left_ind += 1
        print('left inf = ' + str(left_ind))
        right_ind = left_ind+batch_size
        if(i == k_folds - 1):
            right_ind = train_set_x.shape[1] - 1
        print('right inf = ' + str(right_ind))
        test_x = train_set_x[:, left_ind:right_ind]
        test_y = train_set_y[:, left_ind:right_ind]
        aux_x = []
        aux_y = []
        for j in range(0, train_set_x.shape[1] - batch_size):
            aux = right_ind + j
            aux_x.append(train_set_x[:, (aux)%train_set_x.shape[1]:((aux)%train_set_x.shape[1])+1])
            aux_y.append(train_set_y[:, (aux)%train_set_y.shape[1]:((aux)%train_set_y.shape[1])+1])
        train_x = np.array(aux_x).squeeze(axis=2).T
        train_y = np.array(aux_y).squeeze(axis=2).T
        print('test_x shape')
        print(test_x.shape)
        print('test_y shape')
        print(test_y.shape)
        print('train_x shape')
        print(train_x.shape)
        print('train_y shape')
        print(train_y.shape)
        w, b = initialize_with_zeros(train_set_x.shape[0])
        p  = optimize(w, b, train_x, train_y, num_iterations, learning_rate, i + 1)
        Y_prediction_test = predict(w, b, test_x)
        Y_prediction_train = predict(w, b, train_x)
        print("k_fold = " + str(i + 1))
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_y)) * 100))
        print("validation accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_y)) * 100))
        weights += p['w']
        b += p['b']
    weights /= k_folds
    bias /= k_folds
    return weights, bias

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1./(1+np.exp(-z))
    ### END CODE HERE ###
    
    return s


# GRADED FUNCTION: initialize_with_zeros

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros((dim,1))
    b = 0
    ### END CODE HERE ###

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

# GRADED FUNCTION: propagate

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    #print('w')
    #print(w)
    #print(w.shape)
    #print('X')
    #print(X)
    #print(X.shape)
    #print('b')
    #print(b)
    #print('Y')
    #print(Y)
    #print(Y.shape)
    m = X.shape[1]
    #print("m " + str(m))
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A = sigmoid(np.dot(w.T,X) + b)                                    # compute activation
    cost = -1./m*np.sum(Y*np.log(A)+(1.-Y)*np.log(1.-A))                                 # compute cost
    ### END CODE HERE ###
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = 1./m*np.dot(X,(A-Y).T)
    db = 1./m*np.sum(A-Y)
    ### END CODE HERE ###

    #print("dw shape")
    #print(dw.shape)
    #print("w shape")
    #print(w.shape)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

# GRADED FUNCTION: optimize

def optimize(w, b, X, Y, model_id, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    iter = []
    
    for i in range(num_iterations):
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate*dw
        b = b - learning_rate*db
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            iter.append(i)
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    plt.plot(iter, costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.savefig('resources/loss_graph_' + model_id + '.png')
    
    return params

# GRADED FUNCTION: predict

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(w.T,X) + b)
    ### END CODE HERE ###
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        if A[0,i] > .5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
        ### END CODE HERE ###
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, model_id, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    ### START CODE HERE ###
    
    # initialize parameters with zeros (≈ 1 line of code)
    w, b =  initialize_with_zeros(X_train.shape[0])
    #print('w')
    #print(w.shape)
    #print(str(w))
    #print('b')
    #print(str(b))
    
    # Gradient descent (≈ 1 line of code)
    parameters = optimize(w, b, X_train, Y_train, model_id, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    ### END CODE HERE ###
    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
    # Print train/test Errors
    print("train accuracy: {} %".format(train_accuracy))
    print("test accuracy: {} %".format(test_accuracy))

    d = { 
        "w" : w, 
        "b" : b,
        "learning_rate" : learning_rate,
        "num_iterations": num_iterations,
        "train_acc": train_accuracy,
        "test_acc": test_accuracy
        }
    
    return d

def main():
    train_orig_x, train_orig_y, test_orig_x, test_orig_y = load_dataset()

    m_train = train_orig_x.shape[0]
    m_test = test_orig_x.shape[0]
    num_px = train_orig_x.shape[1]

    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Num columns: num_px = " + str(num_px))
    print ("train_set_x shape: " + str(train_orig_x.shape))
    print ("train_set_y shape: " + str(train_orig_y.shape))
    print ("test_set_x shape: " + str(test_orig_x.shape))
    print ("test_set_y shape: " + str(test_orig_y.shape))
    train_x = train_orig_x.reshape(train_orig_x.shape[0], -1).T
    test_x = test_orig_x.reshape(test_orig_x.shape[0], -1).T
    train_y = train_orig_y.reshape((1, train_orig_y.shape[0]))
    test_y = test_orig_y.reshape((1, test_orig_y.shape[0]))
    #k_fold_split(train_x, train_y, 20000, 0.05)
    model_id = str(uuid.uuid4())
    m = model(train_x, train_y, test_x, test_y, model_id, 20000, 0.5)
    print('weights ' + str(m['w']))
    print('bias ' + str(m['b']))
    save_weight_bias(m)
    save_model(m, model_id)

main()