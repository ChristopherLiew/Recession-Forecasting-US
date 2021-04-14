rm(list=ls())

library(lsei)

set.seed(10000)
setwd("C:/Users/yeehs/Desktop/Yee Hsien/NUS/Modules/AY2021 S2/EC4308/Assignments/Project/Results") #set working directory

dp1 = read.csv('DP_Results_1.csv')
dp3 = read.csv('DP_Results_3.csv')
dp6 = read.csv('DP_Results_6.csv')
dp12 = read.csv('DP_Results_12.csv')

rf1 = read.csv('RF_results_1.csv')
rf3 = read.csv('RF_results_3.csv')
rf6 = read.csv('RF_results_6.csv')
rf12 = read.csv('RF_results_12.csv')

xg1 = read.csv('XG_results_1.csv')
xg3 = read.csv('XG_results_3.csv')
xg6 = read.csv('XG_results_6.csv')
xg12 = read.csv('XG_results_12.csv')

# modify the forecasts to be time consistent
dp1 = dp1[-1,]
rf1 = rf1[-1,]

rf6 = rf6[3:nrow(rf6),]
dp6 = dp6[-1,]

# Obtain y test values
y1 = as.matrix(dp1[1])
y3 = as.matrix(dp3[1])
y6 = as.matrix(dp6[1])
y12 = as.matrix(dp12[1])

# remove y test values from data
dp1 = dp1[-1]
dp3 = dp3[-1]
dp6 = dp6[-1]
dp12 = dp12[-1]
xg1 = xg1[-1]
xg3 = xg3[-1]
xg6 = xg6[-1]
xg12 = xg12[-1]
rf1 = rf1[-1]
rf3 = rf3[-1]
rf6 = rf6[-1]
rf12 = rf12[-1]

library(MLmetrics)
# Obtain GR weights for each ensemble
fmatu1 = as.matrix(cbind(dp1, xg1, rf1))
gru1=lsei(fmatu1, as.matrix(y1), c=rep(1,3), d=1, e=diag(3), f=rep(0,3))
View(gru1)

# Obtain combined predictions, log-loss and F1 macro
comb_predu1 = gru1[1]*dp1 + gru1[2]*xg1 + gru1[3]*rf1
LogLoss(as.matrix(comb_predu1), y1)
F1_Score(comb_predu1, y1)

# 3 month forecast
fmatu3 = as.matrix(cbind(dp3, xg3, rf3))
gru3=lsei(fmatu3, as.matrix(y3), c=rep(1,3), d=1, e=diag(3), f=rep(0,3))
View(gru3)

# Obtain combined predictions, log-loss and F1 macro
comb_predu3 = gru3[1]*dp3 + gru3[2]*xg3 + gru3[3]*rf3
LogLoss(as.matrix(comb_predu3), y3)

# 6 month forecast
fmatu6 = as.matrix(cbind(dp6, xg6, rf6))
gru6=lsei(fmatu6, as.matrix(y6), c=rep(1,3), d=1, e=diag(3), f=rep(0,3))
View(gru6)

# Obtain combined predictions, log-loss and F1 macro
comb_predu6 = gru6[1]*dp6 + gru6[2]*xg6 + gru6[3]*rf6
LogLoss(as.matrix(comb_predu6), y6)

# 12 month forecast
fmatu12 = as.matrix(cbind(dp12, xg12, rf12))
gru12=lsei(fmatu12, as.matrix(y12), c=rep(1,3), d=1, e=diag(3), f=rep(0,3))
View(gru12)

# Obtain combined predictions, log-loss and F1 macro
comb_predu12 = gru12[1]*dp12 + gru12[2]*xg12 + gru12[3]*rf12
LogLoss(as.matrix(comb_predu12), y12)

write.csv(cbind(y1, comb_predu1),'ensemble_1.csv')
write.csv(cbind(y3, comb_predu3),'ensemble_3.csv')
write.csv(cbind(y6, comb_predu6),'ensemble_6.csv')
write.csv(cbind(y12, comb_predu12),'ensemble_12.csv')

