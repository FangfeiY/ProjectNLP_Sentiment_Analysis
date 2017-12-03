import numpy as np
import tensorflow as tf
import re

word_vec = np.load('wordVectors.npy')
print('Word vectors loaded!')
wordsList = np.load('wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
print('Word list loaded!')

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

# Text pre-processing
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
def cleanSentences(string):
	string = string.lower().replace("<br />", " ")
	return re.sub(strip_special_chars, "", string.lower())

def getSentenceMatrix(sentence):
	arr = np.zeros([batch_size, max_seq_len])
	sentenceMatrix = np.zeros([batch_size,max_seq_len], dtype='int32')
	cleanedSentence = cleanSentences(sentence)
	split = cleanedSentence.split()
	for indexCounter,word in enumerate(split):
		try:
			sentenceMatrix[0,indexCounter] = wordsList.index(word)
		except ValueError:
			sentenceMatrix[0,indexCounter] = 399999 #Vector for unkown words
	return sentenceMatrix

# Begin demo
inputText = input('Enter your sentence:\n')
inputMatrix = getSentenceMatrix(inputText)
predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
if (predictedSentiment[0] > predictedSentiment[1]):
    print ("Positive Sentiment")
else:
    print ("Negative Sentiment")

