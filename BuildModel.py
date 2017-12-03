import tensorflow as tf
import numpy as np

word_vec = np.load('wordVectors.npy')
print('Word vectors loaded!')

#Set parameters
batch_size = 20
lstm_units = 64
class_num = 2
iterations = 999
max_seq_len = 300
dimension_num = 50

tf.reset_default_graph()

#Construct Tensor
labels = tf.placeholder(tf.float32,[batch_size, class_num])
input_data = tf.placeholder(tf.int32,[batch_size, max_seq_len])

data = tf.Variable(tf.zeros([batch_size, max_seq_len,dimension_num]),dtype = tf.float32)
data = tf.nn.embedding_lookup(word_vec,input_data)

#Feed to LSTM units
lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
lstm_cell = tf.contrib.rnn.DropoutWrapper(cell = lstm_cell,output_keep_prob = 0.75)
value,_ = tf.nn.dynamic_rnn(lstm_cell, data, dtype = tf.float32)

weight = tf.Variable(tf.truncated_normal([lstm_units, class_num]))
bias = tf.Variable(tf.constant(0.1, shape = [class_num]))
value = tf.transpose(value,[1,0,2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last,weight) + bias)

#Calculate Accuracy
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
#print('Accuracy: ' , accuracy)

#Calculate cross entropy loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

#Train the model
print('Start Training')
input_file = input('Enter training data file(in int) name: ')
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
print('Training data loaded and converted to batch array')

input_file = input('Enter label file name: ')
label_list = []
with open(input_file,'r') as input_f:
	for line in input_f:
		label_list.append([float(indicator) for indicator in line.split(',')])
print('Label file loaded')

def GetBatches(iteration,batch_size, max_seq_len):
        label_batch = []
        review_batch = np.zeros([batch_size, max_seq_len])
        for i in range( batch_size):
                review_batch[i] = review_int_arr[iteration * batch_size + i]
                label_batch.append(label_list[iteration * batch_size + i])
        return review_batch, label_batch

print('Begin calculation')

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):
    	#Next Batch of reviews
	nextBatch, nextBatchLabels = GetBatches(i, batch_size, max_seq_len)
	sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
#    #Write summary to Tensorboard
#    if (i % 50 == 0):
#        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
#        writer.add_summary(summary, i)

    #Save the network every 10,000 training iterations
	if (i % 111 == 0 and i != 0):
        	save_path = saver.save(sess, "models/my-model.ckpt", global_step=i)
        	print("saved to %s" % save_path)
# writer.close()
with open('model-out.txt','a') as output_f:
	output_f.write('final prediction: ' + (str)(prediction))
	output_f.write('accuracy: ' + (str)(accuracy))
	output_f.write('correct prediction: ' + (str)(correct_pred))
