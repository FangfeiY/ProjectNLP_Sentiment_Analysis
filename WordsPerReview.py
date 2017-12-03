input_file = input('Enter file name: ')

max = 0
sum = 0
count = 0
with open(input_file) as input_f:
	for line in input_f:
		if len(line.split()) > 1:
			count += 1
			review_length = len(line.split())
			sum = sum + review_length
			if review_length > max:
				max = review_length

print ('max length: ', max)
print ('average length: ', sum/count)
print ('review count: ', count)
