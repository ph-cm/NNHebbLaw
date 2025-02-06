# A regra de Hebb diz que se a saída do i-esimo processador é yi e a ativação d j-ésimo processador é xj então:

#     Wij =  α * Xj * Yi
#     em que α é o tamanho do passo;

# O algoritmo é:
#     0)- inicializar os pesos Wi = 0, onde i = 1, 2, 3 . . .
#         1)- para cada par de treinamento (x,d), faça:
#                 2)- Wi(atual) = Wi(anterior) + αXiDi
#                     Bi(atual) = Bi(anterior) + αDi
#         3)- faça Y* = WiXi + B, onde i = 1, 2, 3 . . .
#     4)- Teste a convergência

import numpy as np

# Input patterns (x1, x2)
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Hebb rule to compute weights and bias
def hebb_rule(inputs, outputs):
    w1, w2, b = 0, 0, 0  # Initialize weights and bias
    for x, y in zip(inputs, outputs):
        w1 += x[0] * y  # Update weight for x1
        w2 += x[1] * y  # Update weight for x2
        b += y          # Update bias
    return w1, w2, b

# Check if the function is linearly separable (in this case, check for XOR and XNOR)
def is_linearly_separable(outputs):
    # XOR and XNOR are not linearly separable
    if outputs == [0, 1, 1, 0] or outputs == [1, 0, 0, 1]:
        return False
    return True

# Function for F0: Constant 0
def F0():
    outputs = [0, 0, 0, 0]
    return hebb_classification(outputs)

# Function for F1: AND
def F1():
    outputs = [0, 0, 0, 1]
    return hebb_classification(outputs)

# Function for F2: Inhibition of x2
def F2():
    outputs = [0, 0, 1, 0]
    return hebb_classification(outputs)

# Function for F3: x1
def F3():
    outputs = [0, 0, 1, 1]
    return hebb_classification(outputs)

# Function for F4: Inhibition of x1
def F4():
    outputs = [0, 1, 0, 0]
    return hebb_classification(outputs)

# Function for F5: x2
def F5():
    outputs = [0, 1, 0, 1]
    return hebb_classification(outputs)

# Function for F6: XOR (Problematic)
def F6():
    outputs = [0, 1, 1, 0]
    return "Error: Function is not linearly separable (XOR)."

# Function for F7: OR
def F7():
    outputs = [0, 1, 1, 1]
    return hebb_classification(outputs)

# Function for F8: NOR
def F8():
    outputs = [1, 0, 0, 0]
    return hebb_classification(outputs)

# Function for F9: XNOR (Problematic)
def F9():
    outputs = [1, 0, 0, 1]
    return "Error: Function is not linearly separable (XNOR)."

# Function for F10: NOT x2
def F10():
    outputs = [1, 0, 1, 0]
    return hebb_classification(outputs)

# Function for F11: Implication x1 -> x2
def F11():
    outputs = [1, 0, 1, 1]
    return hebb_classification(outputs)

# Function for F12: NOT x1
def F12():
    outputs = [1, 1, 0, 0]
    return hebb_classification(outputs)

# Function for F13: Implication x2 -> x1
def F13():
    outputs = [1, 1, 0, 1]
    return hebb_classification(outputs)

# Function for F14: NAND
def F14():
    outputs = [1, 1, 1, 0]
    return hebb_classification(outputs)

# Function for F15: Constant 1
def F15():
    outputs = [1, 1, 1, 1]
    return hebb_classification(outputs)

# Common method to apply the Hebb rule and check linearly separability
def hebb_classification(outputs):
    # Check if the function is linearly separable
    if not is_linearly_separable(outputs):
        return "Error: Function is not linearly separable."
    
    # Apply the Hebb rule to compute weights and bias
    w1, w2, b = hebb_rule(inputs, outputs)
    
    # Check if the computed weights and bias work for all inputs
    correct = True
    for x, y_true in zip(inputs, outputs):
        weighted_sum = w1 * x[0] + w2 * x[1] + b
        y_pred = 1 if weighted_sum >= 0 else 0  # Step function
        if y_pred != y_true:
            correct = False
            break
    
    if correct:
        return "Result: Incorrect classification (not linearly separable)."
    else:
        return f"Weights: w1 = {w1}, w2 = {w2}, Bias = {b}. Result: Correct classification!"

# Call the methods for each function
print(F0())
print(F1())
print(F2())
print(F3())
print(F4())
print(F5())
print(F6())  # This will show an error message for XOR
print(F7())
print(F8())
print(F9())  # This will show an error message for XNOR
print(F10())
print(F11())
print(F12())
print(F13())
print(F14())
print(F15())
