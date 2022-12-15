import numpy as np
import pandas as pd
from apyori import apriori

def inspect(output):
    left = [tuple(result[2][0][0])[0] for result in output]
    support = [result[1] for result in output]
    confidence = [result[2][0][2] for result in output]
    lift = [result[2][0][3] for result in output]
    right = [tuple(result[2][0][1])[0] for result in output]
    return list(zip(left, support, confidence, lift, right))

data = pd.read_csv("Market_Basket_Optimisation.csv",header = None)

transacts = []

for i in range(0, len(data)):
    transacts.append([str(data.values[i,j]) for j in range(0, 20)])
    
rules = apriori(transactions = transacts, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

output = list(rules)
output_data = pd.DataFrame(inspect(output), columns = ['LHS', 'Support', 'Confidence', 'Lift', 'RHS'])
print(output_data)
print(output_data.nlargest(n = 5, columns = "Lift"))
