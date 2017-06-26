import os
import tensorflow as tf

from process_data import process_data
from model import SkipGramModel


def train_flags():
    '''
    Save train configurations
    '''
    tf.app.flags.DEFINE_integer("vocab_size", 50000, "Vocabulary size")
    tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size")
    tf.app.flags.DEFINE_integer("embed_size", 128, "Embedding size")
    tf.app.flags.DEFINE_integer("skip_window", 1, "Contex window")
    tf.app.flags.DEFINE_integer("num_sampled", 64, "Negative sampling number")
    tf.app.flags.DEFINE_integer("num_train_steps", 2000, "Number of training steps")
    tf.app.flags.DEFINE_float("learning_rate", 1.0, "Learning rate")

    tf.app.flags.DEFINE_integer("skip_every", 100, "Output loss every")
    tf.app.flags.DEFINE_bool("restore_from_checkpoint", True, "Start training from the last checkpoint")
    FLAGS = tf.app.flags.FLAGS

    return FLAGS


def train(sess, model, batch_generator, num_train_steps):
    # Initialize a summary writer and a model saver
    writer = tf.summary.FileWriter("graph/vocab_size={}_embed_size={}_contex={}".format(
        FLAGS.vocab_size, FLAGS.embed_size, FLAGS.skip_window), graph=sess.graph)
    saver = tf.train.Saver()

    # Initialize
    total_loss = 0.0
    sess.run(tf.global_variables_initializer())
    
    # If set, retrain from a saved model
    ckpt = tf.train.get_checkpoint_state(os.path.dirname("checkpoints/checkpoint"))
    if FLAGS.restore_from_checkpoint and ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    # Start training
    initial_step = model.global_step.eval()
    for train_step in range(initial_step, initial_step + FLAGS.num_train_steps):
        # Get batch
        centers, targets = next(batch_generator)
        batch = [centers, targets]
        # Update one step
        loss_batch, summary = model.step(batch, sess)
        # Summary this step
        writer.add_summary(summary, train_step)
        # Update loss
        total_loss += loss_batch

        # Print out the loss every few steps
        if (train_step + 1) % FLAGS.skip_every == 0:
            print("Average loss at step {}: {:5.1f}".format(train_step + 1, total_loss / FLAGS.skip_every))
            total_loss = 0.0
    
    # Save the session
    saver.save(sess, "checkpoints/step{}".format(initial_step + FLAGS.num_train_steps),
                 global_step = initial_step + FLAGS.num_train_steps)


if __name__ == '__main__':
    FLAGS = train_flags()
    model = SkipGramModel(FLAGS)

    batch_generator = process_data(FLAGS.vocab_size, FLAGS.batch_size, FLAGS.skip_window)
    with tf.Session() as sess:
        train(sess, model, batch_generator, FLAGS.num_train_steps)
        sess.close()