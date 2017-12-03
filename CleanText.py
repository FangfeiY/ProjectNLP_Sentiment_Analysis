#Argument: input file name 
import re

def clean_str(string):

#Tokenization/string cleaning for all datasets except for SST.
#Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py

	#string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"[^A-Za-z0-9\'\`]+", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"\.+", " . ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " ? ", string)
	string = re.sub(r"\s{2,}", " ", string)

	return string.strip().lower()

input_f_name = input("Enter the file name: ")
cleaned = []
with open (input_f_name,'r') as input_f:
	for line in input_f:
		if line in ['\n','\r\n']:
			cleaned.append(line)
		else:
			cleaned.append(clean_str(line))

output_f_name = "cleaned_" + input_f_name
with open(output_f_name,'a') as output_f:
	for line in cleaned:
		output_f.write(line + '\n')
