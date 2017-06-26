import tensorflow as tf


class SkipGramModel(object):
    '''
    Vannilla Skip-gram model
    ### Ops:
    self.center_words_placeholder
    self.target_words_placeholder
    self.embed_matrix

    self.loss
    self.optimizer

    self.summary_op
    '''

    def __init__(self, flags):
        '''
        Construct and configure the model
        '''
        self._config_model(flags)
        self._build_graph()

    def _get_placeholders(self):
        '''
        ### Build placeholders for input data and labels
            self.center_word_placeholder
            self.target_word_placeholder
        '''
        with tf.name_scope("data"):
            center_words_placeholder = tf.placeholder(
                tf.int32, shape=[self.batch_size], name="center_words")
            target_words_placeholder = tf.placeholder(
                tf.int32, shape=[self.batch_size, 1], name="target_words")

        return center_words_placeholder, target_words_placeholder

    def _get_embeddings(self):
        '''
        ### Build the lookup tabel for word embeddings
            self.embed_matrix
        '''
        with tf.name_scope("embed"):
            embed_matrix = tf.Variable(tf.random_uniform(
                [self.vocab_size, self.embed_size], -1.0, 1.0), name="embed_matrix")

        return embed_matrix

    def _get_loss(self):
        '''
        ### Build the loss op of the graph
        We use Negative Sampling Loss
        '''
        with tf.name_scope("loss"):
            embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words_placeholder, name="embed")

            weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
                                                     stddev=1.0 / (self.embed_size ** 0.5)),
                                 name="nce_weights")
            bias = tf.Variable(
                tf.zeros([self.vocab_size]), name="nce_bias")

            loss = tf.reduce_mean(tf.nn.nce_loss(weights=weight,
                                                 biases=bias,
                                                 labels=self.target_words_placeholder,
                                                 inputs=embed,
                                                 num_sampled=self.num_sampled,
                                                 num_classes=self.vocab_size), name='loss')
        return loss

    def _get_optimizer(self):
        '''
        ### Build the optimizer for training
        We use vannilla batch gradient descent
        '''
        optimizer = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(self.loss)

        return optimizer

    def _get_summaries(self):
        '''
        ### Build the summary ops
        '''
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("loss", self.loss)

            summary_op = tf.summary.merge_all()

        return summary_op

    def _config_model(self, flags):
        '''
        ### Save model configurations
            self.vocab_size
            self.batch_size

            self.embed_size
            self.skip_window
            self.num_sampled

            self.learning_rate
        '''
        self.vocab_size = flags.vocab_size
        self.batch_size = flags.batch_size

        self.embed_size = flags.embed_size
        self.skip_window = flags.skip_window
        self.num_sampled = flags.num_sampled

        self.learning_rate = flags.learning_rate
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

    def _build_graph(self):
        '''
        ### Construct the computation graph
        '''
        self.center_words_placeholder, self.target_words_placeholder = self._get_placeholders()
        self.embed_matrix = self._get_embeddings()
        self.loss = self._get_loss()
        self.optimizer = self._get_optimizer()
        self.summary_op = self._get_summaries()

    def _get_feed_dict(self, batch):
        '''
        Create the feed dictionary given the input train batch
        '''
        feed_dict = {
            self.center_words_placeholder: batch[0],
            self.target_words_placeholder: batch[1]
            }
        
        return feed_dict

    def step(self, batch, sess):
        '''
        ### Perform one train step
        Update the model and return the current batch loss and summary
        '''
        feed_dict = self._get_feed_dict(batch)
        batch_loss, _, summary = sess.run([self.loss, self.optimizer, self.summary_op], feed_dict=feed_dict)

        return batch_loss, summary