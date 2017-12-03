import numpy as np
word_vec = np.load('wordVectors.npy')
print('Word vectors loaded!')
import tensorflow as tf
tf.reset_default_graph()

#Set parameters
batch_size =20 #24
lstm_units = 64
class_num = 2
max_seq_len = 300 #250
dimension_num = 50 #300

labels = tf.placeholder(tf.float32, [batch_size, class_num])
input_data = tf.placeholder(tf.int32, [batch_size, max_seq_len])

data = tf.Variable(tf.zeros([batch_size, max_seq_len, dimension_num]),dtype=tf.float32)
data = tf.nn.embedding_lookup(word_vec,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstm_units, class_num]))
bias = tf.Variable(tf.constant(0.1, shape=[class_num]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

#correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
#accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('my_final_model'))

# Begin Testing
print('Begin testing')
input_file = input('Enter testing data (in int) file: ')
review_int_list = []
with open(input_file, 'r') as input_f:
        for line in input_f:
                if len(line.split()) > 1:
                        review_int_list.append(line)

review_int_arr = np.zeros((len(review_int_list), max_seq_len), dtype = 'int32')
review_counter = 0
for line in review_int_list:
        word_list = line.split()
        if len(word_list) < max_seq_len:
                copy_len = len(word_list)
        else:
                copy_len = max_seq_len
        for i in range(copy_len):
                review_int_arr[review_counter][i] = word_list[i]
        review_counter += 1
print('Testing data loaded and converted to batch array')

input_file = input('Enter gold label file name: ')
gold_label_list = []
with open(input_file,'r') as input_f:
	for line in input_f:
		gold_label_list.append([float(indicator) for indicator in line.split(',')])
print('Gold label file loaded')

#Calculate sentiment and save
pred_label_list = []
current_review_matrix = np.zeros([batch_size,max_seq_len], dtype='int32')
for i in range(len(review_int_list)):
	current_review_matrix[0] = review_int_arr[i]
	pred_sentiment = sess.run(prediction, {input_data: current_review_matrix})[0]
	pred_label_list.append([pred_sentiment[0],pred_sentiment[1]])

#Regulate to 1(positive) and 0(negative)
for pred_label in pred_label_list:
	if pred_label[0] >= 0:
		pred_label[0] = 1
	else:
		pred_label[0] = 0

	if pred_label[1] >= 0:
		pred_label[1] = 1
	else:
		pred_label[1] =0

#Compare with gold
correct = 0
for i in range(len(pred_label_list)):
	if pred_label_list[i][0] == gold_label_list[i][0]:
		correct += 1

accuracy_rate = float(correct) / len(pred_label_list)
print('Accuracy rate: ', accuracy_rate)

