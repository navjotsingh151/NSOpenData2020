import pandas as pd
import numpy as np
import statsmodels.api as sm
# import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression
lm = LinearRegression()

data = pd.read_csv(r"C:\Users\kiran\Downloads\Diseases_per_year.csv")

data.head()
print(data.head())
y = data['Prevalence']
X = data.drop('Prevalence', axis = 1)

def dis(text):
    if text.strip().lower() == 'respiratory':
        return 1
    elif text.strip().lower() == 'cardiovascular':
        return 2
    elif text.strip().lower() == 'diabetes':
        return 3
    elif text.strip().lower() == 'cancer':
        return 4
    else:
        return 0


def age_group(text):
    dict = {'0 to 9': 1,
            '10 to 19': 2,
            '20 to 29': 3,
            '30 to 39': 4,
            '40 to 49': 5,
            '50 to 59': 6,
            '60 to 69': 7,
            '70 to 79': 8,
            '80 +': 9
            }
    return dict.get(text)


data['diseases_int'] = data['diabetes'].apply(lambda x : dis(x))
data['age_int'] = data['Agegroup'].apply(lambda x: age_group(x))

print(data.head())

y = data['Prevalence']
X = data.drop(['Prevalence', 'Agegroup','diabetes'], axis = 1)



# y = data['Prevalance']
lm.fit(X,y)
print(lm.coef_)
print(lm.score(X,y))

