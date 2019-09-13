import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import autopep8
dataset = pd.read_csv("/home/nikhil/Downloads/data.csv")

train_data = dataset.drop(["id","diagnosis","Unnamed: 32"],axis=1)
target = dataset.diagnosis

#===============select population size population x number of attribute=============#
#popln = int(input("please give no. of initial popln:"))
#pc = float(input("probability of crossover:"))
#pm = float(input("probability of mutation:"))
popln = 10 ;pc = 0.7 ; pm = 1/popln ;bits =30

#=====Generate population===============#
pop_binary = []
for i in range(popln):
   binr = np.random.choice([0, 1], size=bits)
   pop_binary.append(binr)

new_data = pd.DataFrame(pop_binary)

#=====IMPORT ML LIBRARY==================#
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,matthews_corrcoef,make_scorer
score = make_scorer(accuracy_score)
model1 = RandomForestClassifier()
column_name = train_data.columns.values
maximum_acc = []
features = []
for i in range(500):
	#============Fitness function===========#
	fitness_score = []
	for i in range(10):
		popl_feature = []
		for j in range(30):
			if new_data.iloc[i][j]==1:
				popl_feature.append(column_name[j])
		new_feaure_space = train_data[popl_feature]
		cv_acc = cross_val_score(estimator=model1,X=new_feaure_space,y=target,cv=5,scoring=score)
		mean_cv_acc = np.mean(cv_acc)
		c = [0.2,0.4,0.6,0.8,1]
		fittest = []
		for k in c:
			fitness = mean_cv_acc - (k*len(popl_feature))/train_data.shape[1]
			fittest.append(fitness)
		fitness_score.append(np.max(fittest))
	max_acc = np.max(fitness_score)
	max_acc_feature = new_data.loc[np.argmax(fitness_score)]
	maximum_acc.append(max_acc)
	features.append(max_acc_feature)
	#================ tournament selection selection==========================#
	fitt = pd.DataFrame(fitness_score)
	selection_data = pd.concat([new_data,fitt],axis=1)
	selection_data = selection_data.rename({0:"fitness"},axis=1)
	select_rand = np.random.randint(0,10,size=(10,2))
	tourn_select = []
	for l in range(len(select_rand)):
		select1 = select_rand[l][0]
		select2 = select_rand[l][1]
		sel1_fit1 = selection_data.iloc[select1,30]
		sel1_fit2 = selection_data.iloc[select2,30]
		if sel1_fit1 > sel1_fit2:
			tourn_select.append(selection_data.iloc[select1])
		else:
			tourn_select.append(selection_data.iloc[select2])
	tourn_df = pd.DataFrame(np.array(tourn_select))
	tourn_df = tourn_df.copy()
	tourn_df1 = tourn_df.drop(30,axis=1).astype(int)
	#=================cross-over==============================================#
	#=====Generate random number=======#
	cross_rand = np.random.randint(0,9,size=(5,2))
	rnd_no = np.random.uniform(0,1)
	cross_prob = 0.6
	cross_df = []
	for m in range((tourn_df1.shape[0])/2):
		cross1 = cross_rand[m][0]
		cross2 = cross_rand[m][1]
		cross_fit1 = tourn_df1.iloc[cross1]
		cross_fit2 = tourn_df1.iloc[cross2]
		for i in range(tourn_df1.shape[1]):
			rnd_no = np.random.uniform(0,1)
			if rnd_no < cross_prob:
				break
		cross_ex1 = cross_fit1.iloc[0:i,].append(cross_fit2.iloc[i:,])
		cross_ex2 = cross_fit2.iloc[0:i,].append(cross_fit1.iloc[i:,])
		cross_df.append(cross_ex1)
		cross_df.append(cross_ex2)
	cross_feature = pd.DataFrame(np.array(cross_df))
	cross_feature1 = cross_feature.copy()
	#================== Mutation Process=========================================#
	mut_prob = 0.3
	for i in range(cross_feature.shape[0]):
		mute_x = cross_feature.loc[i]
		for j in range(cross_feature.shape[1]):
			rnd_no = np.random.uniform(0,1)
			if rnd_no < mut_prob:
				if mute_x[j] == 0:
					mute_x[j] = 1
					break
				else:
					mute_x[j] = 0
					break
	new_data = cross_feature.copy()

feature = pd.DataFrame(features).reset_index()
accuracy = pd.DataFrame(maximum_acc).reset_index()
final_report = pd.concat([feature,accuracy],axis=1)
final_report.drop("index",axis=1,inplace=True)
all_column_name = np.append(column_name,"fitness")
final_report.columns = all_column_name
final_report.to_csv("feat_sel_genetic_code",sep='\t', encoding='utf-8')

	


		


		
		
	

