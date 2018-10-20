import tensorflow as tf
from SingleParticle import SPTrainModel as msm
import os
import time
import numpy as np
from SingleParticle import SPSaveDataSet as sd

def trainSPModel():
    # Parameters for training iteration
    training_epochs = 5000000  #Total Train iterations
    display_step = 1000

    # Parameters for training spectrum
    starter_learning_rate = 0.01

    #Use Stored DataSet
    useDataSet = True

    #Set numpy array display precision
    np.set_printoptions(precision=2)

    # Model Name
    modelName = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\
                +'/SPModel/'\
                +'SPModel'\
                +'.ckpt'

    # Model Path
    logsPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\
                +'/SPModel/'\
                +'logs/'

    # X:input data(spectrum) Y:correct labels
    with tf.name_scope('input'):
        X  = tf.placeholder(tf.float32, shape=(None, msm.n_input), name="Spectrum-input")
        Y  = tf.placeholder(tf.float32, shape=(None,msm.n_output), name="Label-input")

    # Construct model
    trainNet = msm.SPTrainNet()
    trainNet.__init__()

    # Define loss and optimizer
    result, error, loss = trainNet.evaluate(X, Y)

    # Decayed learning rate
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('learning_rate'):
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 3000, 0.96, staircase=True)

    # Define loss and optimizer
    with tf.name_scope('loss'):
        loss_op = tf.reduce_mean(loss)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op, global_step=global_step)

    # create a saver
    saver = tf.train.Saver()

    # Determines the fraction of the overall amount of memory
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    # create a summary for tensorboard
    tf.summary.scalar('total_loss', loss_op)
    tf.summary.scalar('error_R', tf.reduce_mean(error, 0)[0])
    tf.summary.scalar('learning_rate', learning_rate)

    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.summary.merge_all()

    with tf.Session(config=config) as sess:
        # variables need to be initialized before we can use them
        sess.run(tf.global_variables_initializer())

        # create log writer object
        train_writer = tf.summary.FileWriter(logsPath, sess.graph)

        # Try to restore the trained session
        try:
            saver.restore(sess, modelName)
        except:
            saver.save(sess, modelName)

        # Training cycle
        for epoch in range(training_epochs):

            start_time = time.time()

            if useDataSet:
                # Use saved training data set
                data_spectrum, correct_label = sd.loadDataSet()
            else:
                # Use random generate data set
                data_spectrum, correct_label = sd.generateTrainBatches()

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, err, res, summary  = sess.run([train_op, loss_op, error, result, summary_op],
                                                             feed_dict={X: data_spectrum,
                                                                        Y: correct_label})

            train_writer.add_summary(summary, epoch)
            # print("%s ms " % np.int((time.time() - start_time) * 1000))

            # Display logs per epoch step
            if epoch % display_step == 0:

                # Compute average loss
                avg_cost  = c
                avg_errR  = tf.reduce_mean(err, 0)[0]
                avg_errFr = tf.reduce_mean(err, 0)[1]
                avg_errK0 = tf.reduce_mean(err, 0)[2]
                avg_errK1 = tf.reduce_mean(err, 0)[3]

                print("Trained Epoch = %08d " % (global_step.eval(session=sess)-1),
                      "avg_cost = {:.1f}  ".format(avg_cost),
                      "avg_errR = %.1f  " % avg_errR.eval(session=sess),
                      "avg_errFr = %.1f " % avg_errFr.eval(session=sess),
                      "avg_errK0 = %.1f " % avg_errK0.eval(session=sess),
                      "avg_errK1 = %.1f " % avg_errK1.eval(session=sess),
                      "Rate = %.6f " % learning_rate.eval(session=sess),
                      "New Epoch =", '%08d ' % epoch)

                predictR = tf.cast(res[:5, 0], tf.float32)
                print("Predict R = %s " % predictR.eval(session=sess))
                print("correct R = %s  \n" % correct_label[:5, 0])

                predictFR = tf.cast(res[:5, 1], tf.float32)
                print("Predict FR = %s " % predictFR.eval(session=sess))
                print("correct FR = %s  \n" % correct_label[:5, 1])

                predictK0 = tf.cast(res[:5, 2], tf.float32)
                print("Predict K0 = %s " % predictK0.eval(session=sess))
                print("correct K0 = %s  \n" % correct_label[:5, 2])

                predictK1 = tf.cast(res[:5, 3], tf.float32)
                print("Predict K1 = %s " % predictK1.eval(session=sess))
                print("correct K1 = %s  \n" % correct_label[:5, 3])

                saver.save(sess, modelName)

                # If meet the training accuracy, finish training
                if avg_cost < 1:
                    break

        print("Optimization Finished!")


if __name__ == '__main__':

    trainSPModel()
