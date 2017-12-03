import json

input_file = input('Enter file name: ')

amazon_review_list = []
with open(input_file, 'r') as input_f:
	for line in input_f:
		amazon_review_list.append(json.loads(line))

label_list = []
for review in amazon_review_list:
	if (int)(review["overall"]) > 3: #Positive
		label_list.append("1,0")
	else: #Negative
		label_list.append("0,1")

if len(amazon_review_list) ==  len(label_list):
	with open('label-all.txt','a') as output_f:
		for line in label_list:
			output_f.write(line + '\n')
