import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns

df = pd.DataFrame({'Gender' : ['M', 'M', 'M', 'F', 'F'] * 10, 
                    'isSmoker' : ['Smoker', 'Smoker', 'Non-Somker', 'Non-Somker', 'Smoker'] * 10})

c = pd.crosstab(df['Gender'], df['isSmoker'])

sns.heatmap(c, annot = True, cmap = "YlGnBu")

c, p, dof, expected = chi2_contingency(c)

print(p)
