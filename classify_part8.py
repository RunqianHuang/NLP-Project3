import word_embedding as we

p_f = open("../../SentimentDataset/Train/pos_cheat.txt","r")
p_train_text = p_f.readlines()
n_f = open("../../SentimentDataset/Train/neg_cheat.txt","r")
n_train_text = n_f.readlines()

classifier = we.classifier(p_train_text,n_train_text)

test_file = open("../../SentimentDataset/Kaggle/test.txt")
test_text = test_file.readlines()

we_classification = open("../../SentimentDataset/Kaggle/we_result.csv", "w")
we_classification.write("Id,Prediction\n")
n = 1

for line in test_text:
	v = classifier.vectorize(line)
	result = classifier.classify(v)
	we_classification.write(str(n)+","+str(result)+"\n")
	n += 1
	
we_classification.close()
test_file.close()
n_f.close()
p_f.close()
