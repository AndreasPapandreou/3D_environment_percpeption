

filepath = '/home/andreas/Documents/Caramel/caramelav/data/trainingData_part1.dat'
features_updated = ""
keep_inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
ids = ['3', '6']
with open(filepath) as fp:
	for cnt, line in enumerate(fp):
		# if cnt > 4:
			# break

		if cnt == 0:
			line = line.split(" ")
			for i, f in enumerate(line):
				if i in keep_inds:
					if i != len(line)-1:
						features_updated += f + " "
					else:
						features_updated += f
			# features_updated += "\n"
		else:
			new_features = line.split(" ")

			if (new_features[0] in ids):
				# print (new_features[0])
				for i, f in enumerate(new_features):
					if i in keep_inds:
						if i != len(new_features)-1:
							features_updated += f + " "
						else:
							features_updated += f
			# features_updated += "\n"
			# print ("features_updated ", features_updated)


f = open('/home/andreas/Documents/Caramel/caramelav/data/trainingData_part3.dat', "w")
f.write(features_updated)
f.close()









# type
# eigenvaluesSum
# omnivariance 
# eigenentropy 
# linearity 
# planarity 
# sphericity 
# curvatureChange 
# verticalityFirstEigenvectorAxisZ 
# verticalityThirdEigenvectorAxisZ 
# absoluteMomentFirstOrderE1 
# absoluteMomentFirstOrderE2 
# absoluteMomentFirstOrderE3 
# absoluteMomentSecondOrderE1 
# absoluteMomentSecondOrderE2 
# absoluteMomentSecondOrderE3 
# verticalMomentFirstOrder 
# verticalMomentSecondOrder 
# pointsNumber







