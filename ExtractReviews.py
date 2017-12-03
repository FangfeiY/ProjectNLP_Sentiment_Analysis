import json

data_list = []
with open('music_reviews.json','r') as data:
	for line in data:
		data_list.append(json.loads(line))

review_list = []
for review in data_list:
	review_list.append(review["reviewText"])

with open('review-all.txt','a') as output_all:
	for review_text in review_list:
		output_all.write(review_text + '\n\n')
