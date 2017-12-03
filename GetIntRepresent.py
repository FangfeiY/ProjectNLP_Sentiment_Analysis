import numpy as np

wordslist = np.load('wordsList.npy')
wordslist = wordslist.tolist()
wordslist = [word.decode('UTF-8')for word in wordslist]
print('Word list loaded!')


reviews = []
input_file = input('Enter file name: ')
with open(input_file, 'r') as input_f:
	for line in input_f:
		if len(line.split()) > 1:
			reviews.append(line)
print(len(reviews), ' reviews loaded')

max_seq_len = 300
#max_seq_len = len(reviews)
#review_int_list = np.zeros((max_seq_len), dtype = 'int 32')
all_review_int_list = []

for line in reviews:
	review_int = ''
	for word in line.split():
		try:
			#review_int_list[index_counter] = wordslist.index(word)
			review_int += ((str)(wordslist.index(word)) + ' ')
		except ValueError:
			#review_int_list[index_counter] = 399999 #Vector for unknown word
			review_int += '399999 ' #Vector for unknown word

	all_review_int_list.append(review_int)

with open('review-int-out.txt','a') as output_f:
	for review in all_review_int_list:
		output_f.write(review + '\n\n')
