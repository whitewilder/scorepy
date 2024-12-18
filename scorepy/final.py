##########
def main():
	""" Combine functions to make predictions """
	#load data
	train, test = loadDat()
	#make default models
	train, test = runDefaultModels(train, test, 5)
	#make loss models
	train, test = runLossModels(train, test, 5)
	#look at predictions
	print "Percent of Non Defaults:", test.loss.apply(lambda x: x > 0).mean()
	print "Average Loss:", test.loss.mean()
	#save prediction
	test[['id','loss']].to_csv("pred.csv",index=False)

# run everything when calling script from CLI
if __name__ == "__main__":
	main()