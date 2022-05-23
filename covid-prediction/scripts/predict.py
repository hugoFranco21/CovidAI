from time import sleep
from unittest.util import strclass
import numpy as np

def load_weight_bias():
    w = np.load('calc/weights.npy')
    f = open("calc/bias.txt")
    b = float(f.read())
    return w, b

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

def validate_answer(ans):
    message = 'Invalid answer, please enter Y or N'
    if not type(ans) == str:
        print(message)
        sleep(1)
        return 'x'
    if not len(ans) == 1:
        print(message)
        sleep(1)
        return 'x'
    if not ans.upper() == 'Y' and not ans.upper() == 'N':
        print(message)
        sleep(1)
        return 'x'
    return ans.upper()

def convert_ans_num(ans):
    return 0 if ans == 'N' else 1

def cov_message(pred):
    message = 'According to our calculations, you might '
    message += 'not ' if pred[0][0] == 0 else ''
    message += 'have COVID-19. Talk to your doctor regardless of this result.'
    return message

def main():
    w, b = load_weight_bias()
    flag = True
    while flag:
        bp = f = dc = st = at = contact = lgath = fam = 0
        print('Welcome to the COVID-19 NN predictor, answer Y or N if you have had \
the following symptoms:')
        ans = 'x'
        while not ans == 'Y' and not ans == 'N':
            ans = input('Breathing problems: ')
            ans = validate_answer(ans)
        bp = convert_ans_num(ans)
        ans = 'x'
        while not ans == 'Y' and not ans == 'N':
            ans = input('Fever: ')
            ans = validate_answer(ans)
        f = convert_ans_num(ans)
        ans = 'x'
        while not ans == 'Y' and not ans == 'N':
            ans = input('Dry cough: ')
            ans = validate_answer(ans)
        dc = convert_ans_num(ans)
        ans = 'x'
        while not ans == 'Y' and not ans == 'N':
            ans = input('Sore throat: ')
            ans = validate_answer(ans)
        st = convert_ans_num(ans)
        print('Now, answer Y and N if any of these apply to you')
        ans = 'x'
        while not ans == 'Y' and not ans == 'N':
            ans = input('Did you travel abroad in the past 14 days: ')
            ans = validate_answer(ans)
        at = convert_ans_num(ans)
        ans = 'x'
        while not ans == 'Y' and not ans == 'N':
            ans = input('Did you come into contact with a COVID-19 patient in the past 14 days: ')
            ans = validate_answer(ans)
        contact = convert_ans_num(ans)
        ans = 'x'
        while not ans == 'Y' and not ans == 'N':
            ans = input('Did you attend a large gathering in the past 14 days: ')
            ans = validate_answer(ans)
        lgath = convert_ans_num(ans)
        ans = 'x'
        while not ans == 'Y' and not ans == 'N':
            ans = input('Do you have a family member currently working in a public or exposed place: ')
            ans = validate_answer(ans)
        fam = convert_ans_num(ans)
        elem = [[bp, f, dc, st, at, contact, lgath, fam]]
        X_p = np.asarray(elem)
        X = np.reshape(X_p, (X_p.shape[1],X_p.shape[0]))
        print('Calculating...')
        sleep(2)
        pred = predict(w,b,X)
        print(cov_message(pred))
        sleep(2)
        ans = 'x'
        while not ans == 'Y' and not ans == 'N':
            ans = input('Would you like another prediction: ')
            ans = validate_answer(ans)
        if ans == 'N':
            flag = False
        else:
            print('\n')

main()