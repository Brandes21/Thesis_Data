import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


file_path = 'stress_data_2.xlsx'
df = pd.read_excel(file_path)

# Extract the stress data for each design
data_design1 = df['Design1']
data_design2 = df['Design2']

# Calculate mean and standard deviation for each design
mean1, std1 = data_design1.mean(), data_design1.std()
mean2, std2 = data_design2.mean(), data_design2.std()

# Create x values for plotting the normal distributions.
x1 = np.linspace(mean1 - 4*std1, mean1 + 4*std1, 1800)
x2 = np.linspace(mean2 - 5*std2, mean2 + 5*std2, 1800)

# Compute the probability density functions (bell curves)
pdf1 = norm.pdf(x1, mean1, std1)
pdf2 = norm.pdf(x2, mean2, std2)

# Plot both bell curves
plt.figure(figsize=(10,6))
plt.plot(x1, pdf1, label='Equal', color='blue')
plt.plot(x2, pdf2, label='Force-based', color='red')
plt.title('Bell curves of stress values for Layout 1 and 2')
plt.xlabel('Stress Values')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()


# import numpy as np 
# import matplotlib.pyplot as plt 
  
# # A custom function to calculate 
# # probability distribution function 
# def pdf(x): 
#     mean = np.mean(x) 
#     std = np.std(x) 
#     y_out = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2)) 
#     return y_out 
    

# x = df
  
# y = pdf(x) 
  
# # Plotting 
# plt.style.use('seaborn') 
# plt.figure(figsize = (6, 6)) 
# plt.plot(x, y, color = 'black', 
#          linestyle = 'dashed') 
  
# plt.scatter( x, y, marker = 'o', s = 25, color = 'red') 
# plt.show()