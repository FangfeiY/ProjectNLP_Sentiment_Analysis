import json

data_list = []
with open('music_reviews.json','r') as data:
	for line in data:
		data_list.append(json.loads(line))

above_25_list = []
below_25_list = []

for review in data_list:
	if (int)(review["overall"]) >= 2.5:
		above_25_list.append(review["reviewText"])
	else:
		below_25_list.append(review["reviewText"])

with open('above_25.txt','a') as output_above:
	for review_text in above_25_list:
		output_above.write(review_text + '\n\n')

with open('below_25.txt','a') as output_below:
	for review_text in below_25_list:
		output_below.write(review_text + '\n\n')
