from numpy import sort

test = [('AMB_TEMP', 0.058356866), ('CO', 0.3404203), 
('NO', 0.014487224), ('NO2', 0.039495062), ('NOx', 0.010667064), 
('O3', 0.1419492), ('RH', 0.02899634), ('SO2', 0.24512722), 
('TIME', 0.033013362), ('WD_HR', 0.013408621), ('WIND_DIREC', 0.0072317747), 
('WIND_SPEED', 0.023302361), ('WS_HR', 0.04354463)]
a = sorted(test, key=lambda x: x[1])
b = float('inf')
print(1<b)
b = 0
print(1<0)

newDict = [feature for (feature,threshold) in a if threshold > 0.04]
print(newDict)