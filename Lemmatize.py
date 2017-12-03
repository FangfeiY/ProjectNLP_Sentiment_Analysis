from nltk.stem import WordNetLemmatizer
input_file = input('Enter file name: ')

all_lmtz_list = []
#lmtz_list = []
review_lmtz = ''
lemmatizer = WordNetLemmatizer()

with open(input_file, 'r') as input_f:
	for line in input_f:
		if len(line.split()) > 1:
			for word in line.split():
				#lmtz_list.append(lemmatizer.lemmatize(word))0
				review_lmtz += (lemmatizer.lemmatize(word)+ ' ')
		else:
			if len(review_lmtz.split()) > 1:
				all_lmtz_list.append(review_lmtz)
			#lmtz_list = []
			review_lmtz = ''
with open('lmtz-out.txt','a') as output_f:
	for review in all_lmtz_list:
		output_f.write(review + '\n\n')
