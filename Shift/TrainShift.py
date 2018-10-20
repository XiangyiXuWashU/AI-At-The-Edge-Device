import tensorflow as tf
from Shift import ModeShift as ms, ModeShiftTrainModel as msm
import numpy as np
import os

# Parameters for training iteration
training_epochs = 10000  #Total Train iterations
batch_size = 500         #Train iterations for each random deltaT
display_step = 10

# Parameters for training spectrum
shift_range = 1.8
train_k0 = 20e+6
train_k1 = 20e+6
train_startFreq = -2
train_stopFreq = 2


# X:input data(spectrum) Y:correct labels(shift value)
X = tf.placeholder(tf.float32, shape=(None, msm.n_input))
Y = tf.placeholder(tf.float32)

# Construct model
trainNet = msm.ModeShiftTrainNet()
trainNet.__init__()

# Define loss and optimizer
result, loss, error = trainNet.evaluate(X,Y)

# Ｄecayed learning rate
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           batch_size*100, 0.96, staircase=True)

# Define loss and optimizer
loss_op = tf.reduce_mean(error)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op, global_step=global_step)

# Create a ModeShift instance
resonance = ms.ModeShift()
resonance.__init__(k0=train_k0, k1=train_k1)

# create a saver
saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

# Determines the fraction of the overall amount of memory
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8


# Model Name
modelName = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\
            +'/ShiftModel/'\
            +'shiftModel'\
            +'_k0:%.1e'%train_k0 \
            +'_k1:%.1e'%train_k1 \
            +'_start:%.1f'%train_startFreq \
            +'_stop:%.1f'%train_stopFreq \
            +'_number:%.0f'%msm.n_input \
            +'.ckpt'


with tf.Session(config=config) as sess:
    sess.run(init)
    # Try to restore the trained session
    try:
        saver.restore(sess, modelName)
    except:
        saver.save(sess, modelName)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.

        # Loop over all batches
        for i in range(batch_size):
            # Generate random mode shift
            random_DeltaT = np.random.uniform(-shift_range , shift_range)

            # Generate spectrum
            _, _, spectrumAfter = resonance.calculateSpectrum(
                startFreq=train_startFreq, stopFreq=train_stopFreq,
                number=msm.n_input, deltaT=random_DeltaT)

            data_spectrum = np.asarray([spectrumAfter], np.float32)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: data_spectrum,
                                                            Y: random_DeltaT})
            # Compute average loss
            avg_cost += c / batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Trained Epoch = %06d   " % (global_step.eval(session=sess)/batch_size-1),
                  "cost = {:.6f} GHz   ".format(avg_cost),
                  "Learning Rate = %.6f   " %learning_rate.eval(session=sess),
                  "New Epoch =", '%06d   ' % epoch)

        saver.save(sess, modelName)

        # If meet the training accuracy, finish training
        if avg_cost < 0.001:
            break

    print("Optimization Finished!")
