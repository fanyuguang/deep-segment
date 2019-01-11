import tensorflow as tf


# folder path
tf.app.flags.DEFINE_string('raw_data_path', 'data/raw-data/', 'raw data directory')
tf.app.flags.DEFINE_string('datasets_path', 'data/datasets/', 'datasets directory')
tf.app.flags.DEFINE_string('vocab_path', 'data/vocab/', 'vocab directory')
tf.app.flags.DEFINE_string('tfrecords_path', 'data/tfrecords/', 'tfrecords directory')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoint/', 'checkpoint directory')
tf.app.flags.DEFINE_string('tensorboard_path', 'tensorboard/', 'tensorboard directory')
tf.app.flags.DEFINE_string('freeze_graph_path', 'freeze-graph-data', 'frozen graph directory')
tf.app.flags.DEFINE_string('saved_model_path', 'saved-model-data', 'saved model and variables directory, for tensorflow serving')

# training data process params
tf.app.flags.DEFINE_bool('use_char_based', True, 'if True, model is char-based, else segmentation word-based')
tf.app.flags.DEFINE_string('default_label', 'WORD-S', 'define the default label in the label_vocab')
tf.app.flags.DEFINE_integer('vocab_size', 8000, 'vocab size')
tf.app.flags.DEFINE_float('train_percent', 0.8, 'train percent')
tf.app.flags.DEFINE_float('val_percent', 0.1, 'val test percent')

# batch data generator params
tf.app.flags.DEFINE_integer('batch_size', 100, 'words batch size')
tf.app.flags.DEFINE_integer('min_after_dequeue', 10000, 'min after dequeue')
tf.app.flags.DEFINE_integer('num_threads', 1, 'read batch num threads')
tf.app.flags.DEFINE_integer('num_steps', 200, 'num steps, equals the length of words')

# model params
tf.app.flags.DEFINE_bool('use_stored_embedding', False, 'if True, using pretrained word embedding, else random initialize word embedding')
tf.app.flags.DEFINE_bool('use_lstm', True, 'if True, using lstm, else using gru')
tf.app.flags.DEFINE_bool('use_crf', True, 'if True, model structure lstm+crf, else model structure lstm')
tf.app.flags.DEFINE_bool('use_dynamic_rnn', True, 'if True, using dynamic lstm, else using static lstm with fixed setnence length')
tf.app.flags.DEFINE_bool('use_bidirectional_rnn', True, 'if True, using bidirectional rnn, else using forward rnn')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.9, 'learning rate decay factor')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'decay steps')
tf.app.flags.DEFINE_integer('num_layers', 2, 'lstm layers')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'word embedding size')
tf.app.flags.DEFINE_integer('hidden_size', 100, 'lstm hidden size')
tf.app.flags.DEFINE_float('keep_prob', 0.5, 'keep prob')
tf.app.flags.DEFINE_float('clip_norm', 5.0, 'clipping ratio')
tf.app.flags.DEFINE_integer('max_training_step', 200000, 'max training step')

tf.app.flags.DEFINE_float('default_predict_score', 0.0, 'define the default label in the label_vocab')

FLAGS = tf.app.flags.FLAGS
