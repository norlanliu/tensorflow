#!/usr/bin/env python
"""
A Tensorflow implementation of Character-Aware Language Models.

Paper can be found at: http://arxiv.org/pdf/1508.06615v4.pdf

For information about how to run the model training, run

    $ python main.py --help

This model serves as a benchmark for testing the MPI allreduce ops in a
real-world setting.

The model is a language model which predicts the next word in a sentence given
all previous words. Every character is converted to a low-dimensional
embedding; words then are a series of character embeddings. Every word is
processed using convolutional layers with varying widths (usually between width
1 and width 7); the outputs of the different convolution layers are
concatenated and form a single vector. In order to handle variable length
words, words are padded to a maximum word length, and after the convolutions,
the activations are maxpooled over the word length. The vectors then go through
a few fully connected layers, and the output of these fully connected vectors
are word embeddings. These word embeddings are then fed into a multilayer GRU
RNN. The outputs of the RNN are finally fed into a fully connected softmax
layer, with the softmax size being equal to the vocabulary size (plus one for
the UNK token).

The model is then optimized with the Adam optimization algorithm, running in
parallel using MPI on many GPUs (with one GPU per MPI processes) and using a
ring allreduce for gradient averaging. To do this, every process must process a
different subset of the data, which is done by choosing a subset which is a
function of the MPI rank of the process.
"""
from collections import namedtuple
import os
import time
import math

import random
random.seed(10234)

import click

import numpy as np

import tensorflow as tf
import tensorflow.contrib.mpi as mpi

# Special symbols included in the vocabulary
START_SYMBOL = "<S>"
END_SYMBOL = "</S>"
UNK_SYMBOL = "<UNK>"

# Characters allowed in input words. Other characters are mapped to UNK_CHAR.
CHARS = "".join([
    '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~',
    '0123456789',
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
    'abcdefghijklmnopqrstuvwxyz',
])

# Special characters used for the character CNN
# NO_CHAR:    Padding character for words shoter than max word length
# START_CHAR: Start character before the word
# END_CHAR:   End character after the word, before padding
# UNK_CHAR:   An unknown character (including all non-ASCII)
NO_CHAR = 0
START_CHAR = 1
END_CHAR = 2
UNK_CHAR = 3

# A mapping of characters to integers.
# Start at 4 because 0-3 are reserved for NO_CHAR, START_CHAR, etc.
CHAR_MAP = dict((char, i + 4) for (i, char) in enumerate(CHARS))

# Total number of characters (dimensionality of input layer to CNN).
NUM_CHARS = 4 + len(CHARS)

# Container for models.
#   samples: placeholder for inputs
#   labels:  placeholder for labels
#   loss:    cross-entropy loss computed from samples and labels
Model = namedtuple("Model", "samples labels loss")

# List of GPUs available to this process. When using MPI, only one GPU is used
# regardless of number of visible GPUs.
NUM_GPUS = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
GPUS = ["/gpu:{}".format(gpu) for gpu in range(NUM_GPUS)]

# Get rank, local rank, and size from the environment.
# This will need to be modified on different clusters and MPI implementations.
MPI_RANK = int(os.environ["PMI_RANK"])
MPI_LOCAL_RANK = int(os.environ["SLURM_LOCALID"])
MPI_SIZE = int(os.environ["PMI_SIZE"])


def load_vocab(filename, vocab_size):
    """
    Load vocabulary from a vocabulary file.

    The vocabulary file must contain one word per line, sorted with the most
    common words at the top. In addition to the vocabulary in the file, the
    vocabulary returned will contain symbols for <UNK>, <S>, and </S>, for
    unknown words, the start of a sentence, and the end of a sentence,
    respectively. The returned vocabulary is a hashmap from the vocabulary
    word to an integer index for the word.
    """
    vocab = {}
    vocab[UNK_SYMBOL] = 0
    vocab[START_SYMBOL] = 1
    vocab[END_SYMBOL] = 2

    seen = len(vocab)
    with open(filename, "r") as handle:
        for line in handle:
            word = line.strip()
            if word not in vocab:
                vocab[word] = seen
                seen += 1

            # Stop once we've added as many vocab words as we want
            if len(vocab) == vocab_size:
                break
    return vocab


def encode_word(word, max_word_length, vocab):
    """
    Encode a word in a format suitable for the Char-RNN.

    Specifically, each word goes through the following process:
        1. If the word is not in the vocabulary, it is replaced by UNK_SYMBOL.
        2. If the word is longer than the maximum word length minus two, it is
           trimmed to a substring (starting at the beginning of the word). The
           length is word length minus to two account for the start and end
           characters.
        3. Every character is replaced with the integer that corresponds to it;
           unknown characters are replaced with UNK_CHAR.
        4. The list of character indices is prefixed with START_CHAR and
           suffixed with END_CHAR, to indicate the beginning and end of the
           word.
        5. The list of character indices is padded (at the end) with as many
           copies of NO_CHAR as necessary to generate a maximum-word-length
           vector.

    The final list of character indices is returned.
    """
    if word not in vocab:
        word = UNK_SYMBOL

    # Cut down word to maximum allowed length
    if len(word) > max_word_length - 2:
        word = word[:max_word_length - 2]

    # Encode a word as the start character, followed by the indices
    # of each of the characters in the word, followed by the end character,
    # followed by as many empty characters as needed to create a
    # full-length (max word length) vector. Replace unknown characters
    # (mostly Unicode) with a fixed character.
    return ([START_CHAR] +
            [CHAR_MAP.get(char, UNK_CHAR) for char in word] +
            [END_CHAR] +
            [NO_CHAR] * (max_word_length - len(word) - 2))


def batches_from(filename, vocab, batch_size, max_word_length, sequence_length,
                 worker_index, worker_count):
    """
    A generator that yields an infinite series of batches from the provided
    file, using the provided maximum word length, sequence length, and batch
    size.

    The returned value is a tuple (samples, labels). `samples` is a numpy
    array of dimension batch_size X sequence_length X max_word_length, where
    each value is a character index (for input to a char RNN); `labels` is a
    numpy array of dimension batch_size x sequence_length, where each value is
    a word index in the vocabulary.

    The label for every timestep corresponds to the word at the *next* position
    in the sentence. To do this, the file is read in sequence_length + 1
    chunks, where the first sequence_length words in a chunk are used to make
    the sample output, and the last sequence_length words are used to make the
    label output.

    This function also takes `worker_index` and `worker_count` parameters,
    which can be used when you have multiple worker threads or processes
    reading from the same data file. When these are not None, the size of the
    file is computed, and it is broken into `worker_count` chunks. The returned
    generator reads from the `worker_index` chunk; each chunk is guaranteed to
    start at a newline.
    """
    unk_word = vocab[UNK_SYMBOL]
    num_words = sequence_length + 1

    encoded = []
    words = []

    file_size = os.path.getsize(filename)
    chunk_size = math.ceil(file_size / worker_count)
    file_start = worker_index * chunk_size
    file_end = (worker_index + 1) * chunk_size

    while True:
        with open(filename, "r") as handle:
            # Seek to the starting point and find a newline.
            handle.seek(file_start)
            handle.readline()

            sentence = handle.readline()
            while sentence != '' and handle.tell() < file_end:
                # Encode all words in the sentence into char RNN format
                sent_words = [START_SYMBOL] + sentence.split() + [END_SYMBOL]
                encoded.extend(encode_word(word, max_word_length, vocab)
                               for word in sent_words)

                # Encode all words in the sentence in label format
                words.extend(vocab.get(word, unk_word) for word in sent_words)

                # If we have enough words to yield the next batch, do so.
                if len(words) >= batch_size * num_words:
                    samples = []
                    labels = []
                    last_idx = None
                    for i in range(0, batch_size * num_words, num_words):
                        samples.append(encoded[i:i + sequence_length])
                        labels.append(words[i + 1: i + sequence_length + 1])
                        last_idx = i

                    # Discard samples and labels we're about to yield
                    words = words[last_idx:]
                    encoded = encoded[last_idx:]

                    yield (np.asarray(samples, dtype=np.uint8),
                           np.asarray(labels, dtype=np.uint8))
                sentence = handle.readline()


def build_char_embedding(input_seq, embedding_size):
    """Perform character embedding with a tf.gather."""
    initializer = tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32)

    # Embedding Weights: |Chars| x EmbeddingDim
    W_embedding = tf.get_variable("EmbeddingWeights",
                                  shape=(NUM_CHARS, embedding_size),
                                  dtype=tf.float32, initializer=initializer)

    # Embedding: BatchSize X SeqLength X WordLength X EmbeddingDim
    char_embeddings = tf.gather(W_embedding, input_seq, name="CharEmbedding")

    return char_embeddings


def build_convolution_layers(convolution_layers, char_embeddings):
    """Perform a series of convolutions of varying widths. Then, for each
    filter, select the maximum activation across the word; this corresponds to
    a maxpool of the same dimension as the output.

    Return the concatenated output from the convolutions of all widths.
    """
    initializer = tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32)

    embedding_size = char_embeddings.get_shape()[-1].value
    max_word_length = char_embeddings.get_shape()[-2].value

    convolutions = []
    for conv_width, num_filters in convolution_layers.items():
        with tf.variable_scope("ConvWidth-{}".format(conv_width)):
            # Convolution Filters: 1 X FilterWidth X EmbeddingDim X NumFilters
            #
            # The one is necessary in front because we are not convolving over
            # different elements of the batch.
            W = tf.get_variable(
                "Filters", shape=(1, conv_width, embedding_size, num_filters),
                dtype=tf.float32, initializer=initializer)

            # Convolutions: BatchSize X SeqLength X OutWidth X NumFilters
            #
            # For each convolution we have a separate FilterWidth and a
            # configurable NumFilters. The OutWidth is given by:
            #     OutWidth = WordLength - FilterWidth + 1
            convolved = tf.nn.conv2d(char_embeddings, W,
                                     strides=[1, 1, 1, 1], padding="VALID")
            out_width = max_word_length - conv_width + 1

            # Pooling: BatchSize X SeqLength X 1 X NumFilters
            #
            # The pooling gets rid of the OutWidth by pooling over all the
            # points output by the filter.
            pooled = tf.nn.max_pool(convolved, ksize=[1, 1, out_width, 1],
                                    strides=[1, 1, 1, 1], padding="VALID")

            # Squeeze: BatchSize X SeqLength X NumFilters
            squeezed = tf.squeeze(pooled, squeeze_dims=[2], name="SqueezePool")
            convolutions.append(squeezed)

    # Full Convolution Output: BatchSize X SeqLength X Sum(NumFilters)
    full_conv = tf.tanh(tf.concat(2, convolutions), name="WordEmbedding")

    return full_conv


def build_feedforward_layers(layer_sizes, layer_input):
    """Perform feedforward propagation, generating the word embeddings for the
    Char-CNN; uses either highway or fully-connected layers."""
    input_dim = layer_input.get_shape()[-1].value
    initializer = tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32)

    for i, layer_size in enumerate(layer_sizes):
        with tf.variable_scope("FullyConnected-{0}".format(i)):
            W = tf.get_variable("Weights", shape=(input_dim, layer_size),
                                dtype=tf.float32, initializer=initializer)
            b = tf.get_variable("Bias", shape=(1, layer_size),
                                dtype=tf.float32, initializer=initializer)
            layer_input = tf.tanh(tf.matmul(layer_input, W) + b)
            input_dim = layer_size

    return layer_input


def build_recurrent_layers(layer_sizes, layer_input):
    """Perform the recurrent layer update steps for all time."""
    # Recurrent Layers:
    #   Final State: (BatchSize X RecurrentSize, BatchSize X RecurrentSize)
    #   Timesteps: BatchSize X RecurrentSize
    for i, recurrent_size in enumerate(layer_sizes):
        with tf.variable_scope("Recurrent-{}".format(i)):
            cell = tf.nn.rnn_cell.GRUCell(recurrent_size)
            layer_input, _ = tf.nn.dynamic_rnn(
                cell, layer_input, dtype=tf.float32)

    # Concat: (BatchSize * SeqLength) X LastRecurrentSize
    #
    # Concatenate all outputs into a single vector. Instead of outputting a
    # 3-tensor with the batch size and sequence length as two separate
    # dimensions, we output a 2-tensor with one dimension that is both batch
    # size and sequence length. This makes it easier to do later softmax
    # layers, as it requires only one matrix multiply, rather than one per
    # timestep.
    output = tf.reshape(layer_input, [-1, recurrent_size])

    return output


def build_softmax(layer_input, vocab_size):
    """Perform softmax (or dsoftmax) for all output timesteps."""
    input_dim = layer_input.get_shape()[-1].value
    initializer = tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32)

    # Logit:  (BatchSize * SeqLength) X |Vocab|
    with tf.variable_scope("Softmax"):
        W = tf.get_variable("Weight", shape=(input_dim, vocab_size),
                            dtype=tf.float32, initializer=initializer)
        b = tf.get_variable("Bias", shape=(1, vocab_size),
                            dtype=tf.float32, initializer=initializer)
        logits = tf.matmul(layer_input, W) + b

    return logits


def build_network(sequence_length, max_word_length, vocab_size, embedding_size,
                  convolution_layers, feedforward_layers, recurrent_layers):
    """
    Create our language model.

    Most of the variable parameters should be loaded from a model config using
    `load_config`, then passed to this function as keyword arguments.
    """
    tf.set_random_seed(1234)

    with tf.name_scope("Embedding"):
        # Input (int): BatchSize X SeqLength X WordLength
        input_shape = [None, sequence_length, max_word_length]
        input_seq = tf.placeholder(tf.int32, shape=input_shape, name="Input")
        char_embeddings = build_char_embedding(input_seq, embedding_size)

    with tf.name_scope("Convolution"):
        convolutions = build_convolution_layers(
            convolution_layers, char_embeddings)
        convolution_dim = sum(convolution_layers.values())

    # Feed-forward Layers: BatchSize X SeqLength X RecurrentInputDim
    with tf.name_scope("Feedforward"):
        layer = tf.reshape(convolutions, [-1, convolution_dim],
                           name="RemoveTimeDimension")
        layer = build_feedforward_layers(
            feedforward_layers, layer)
        recurrent_input = tf.reshape(
            layer, [-1, sequence_length, feedforward_layers[-1]],
            name="ReshapeForRNN")

    with tf.name_scope("Recurrent"):
        rnn_output = build_recurrent_layers(recurrent_layers, recurrent_input)

    with tf.name_scope("Softmax"):
        logits = build_softmax(rnn_output, vocab_size)

    with tf.name_scope("Loss"):
        labels = tf.placeholder(tf.int64,
                                shape=[None, sequence_length],
                                name="Labels")
        labels_flat = tf.reshape(labels, [-1], name="Flatten")

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels_flat)
        loss_aggregate = tf.reduce_mean(loss, name="CrossEntropyMean")

    return (input_seq, labels), loss_aggregate


def take(n, sequence):
    """Return the first `n` elements in a sequence as a list."""
    return [value for (value, _) in zip(sequence, range(n))]


def parameter_count():
    """Return the total number of parameters in all Tensorflow-defined
    variables, using `tf.all_variables()` to get the list of variables."""
    return sum(np.product(var.get_shape().as_list())
               for var in tf.all_variables())


def clip_gradient(gradient, max_gradient_norm):
    """Clip a gradient to a maximum norm. If the norm of the gradient is
    greater than the provided maximum norm, rescale the gradient to be the
    maximum norm.

    The gradients are expected to be a list of (gradient, variable) pairs.

    Return the new clipped gradients and the original gradient norm.
    """
    # Unpack the gradients to get rid of the variables.
    gradient_tensors = [grad for (grad, _) in gradient]

    clipped_tensors, grad_norm = tf.clip_by_global_norm(gradient_tensors,
                                                        max_gradient_norm,
                                                        name="GradClipNorm")

    # Pack the now-clipped gradients back up with their respective variables.
    clipped_gradient = [(clipped, var)
                        for (clipped, (_, var))
                        in zip(clipped_tensors, gradient)]

    return clipped_gradient, grad_norm


def train_model(session, train_batches, validation_batches, model_inputs,
                train_step, loss, iterations):
    """Run training on the model.

    Specifically, for each iteration up to the maximum number of iterations,
    obtain the next set of training samples from the `train_batches` batch
    generator, run `run_train`, and output timing statistics. Every
    `test_every` steps, use `run_forward` to get loss on a validation set
    obtained from `validation_batches`, and every `save_every` steps, save the
    model using the provided `saver`.
    """
    # Collect validation inputs for each GPU. We have a single validation
    # dataset, so we don't need to repeat this on every iteration.
    valid_inputs = {}
    for batch, model_input in zip(validation_batches, model_inputs):
        input_seq, labels = model_input
        batch_inputs, batch_labels = batch
        valid_inputs.update({
            input_seq: batch_inputs,
            labels: batch_labels,
        })

    for step in range(iterations):
        # Collect inputs for every GPU
        inputs = {}
        for input_seq, labels in model_inputs:
            batch_inputs, batch_labels = next(train_batches)
            inputs.update({
                input_seq: batch_inputs,
                labels: batch_labels,
            })

        # Run training step and record time.
        start_time = time.time()
        _, train_loss = session.run([train_step, loss], feed_dict=inputs)
        elapsed = time.time() - start_time

        if step == 0:
            global_start = time.time()
        global_elapsed = time.time() - global_start

        # Print time taken; only do so on MPI rank 0 if using MPI.
        if MPI_RANK == 0:
            print("Elapsed: \t{:.3f} – Step: \t{:<5} \tTime: \t{:.3f}"
                  .format(global_elapsed, step + 1, elapsed), flush=True)


@click.command()
@click.option("--train-data", help="train.txt", required=True)
@click.option("--validation-data", help="validation.txt", required=True)
@click.option("--vocab", help="vocab.txt", required=True)
@click.option("--vocab-size", help="Vocab size", default=10000)
@click.option("--batch-size", help="Batch size", default=256)
@click.option("--max-word-length",
              help="Maximum word length (longer words are clipped)",
              default=40)
@click.option("--sequence-length",
              help="Number of words to put into every sequence.",
              default=20)
@click.option("--max-iterations",
              help="Maximum number of iterations to run",
              default=100000)
@click.option("--max-gradient-norm", default=10000.0,
              help="Max gradient norm to clip to")
@click.option("--embedding-size", default=15,
              help="Dimension of the character embedding")
@click.option("--convolution-sizes", default="256,512,768,768,768,512,256",
              help="Comma-separated numbers of filters per convolution width")
@click.option("--feedforward-sizes", default="1024,1024",
              help="Comma-separated widths for feedforward layers")
@click.option("--recurrent-sizes", default="2048,2048,2048",
              help="Comma-separated widths for recurrent GRU layers")
def main(train_data, validation_data, batch_size, max_word_length,
         sequence_length, vocab, vocab_size, max_iterations, max_gradient_norm,
         embedding_size, convolution_sizes, recurrent_sizes,
         feedforward_sizes):

    print("Detected GPUs: {}.".format(len(GPUS)), flush=True)

    # Load the vocabulary into memory. The model is automatically initialized
    # to handle the loaded vocabulary size.
    vocab = load_vocab(vocab, vocab_size)

    train_batches = batches_from(train_data, vocab, batch_size,
                                 max_word_length, sequence_length,
                                 MPI_RANK, MPI_SIZE)
    validation_batches = take(len(GPUS),
                              batches_from(validation_data, vocab, batch_size,
                                           max_word_length, sequence_length,
                                           MPI_RANK, MPI_SIZE))
    convolution_sizes = dict(
        ((i + 1, int(x)) for i, x in enumerate(convolution_sizes.split(","))))
    recurrent_sizes = [int(x) for x in recurrent_sizes.split(",")]
    feedforward_sizes = [int(x) for x in feedforward_sizes.split(",")]

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    inputs, loss = build_network(
        sequence_length, max_word_length, vocab_size, embedding_size,
        convolution_sizes, feedforward_sizes, recurrent_sizes)
    optimizer = mpi.DistributedOptimizer(optimizer)
    gradients = optimizer.compute_gradients(loss)
    gradient, grad_norm = clip_gradient(gradients, max_gradient_norm)
    train_step = optimizer.apply_gradients(gradient)

    # Compute total number of parameters (for debugging).
    total_params = parameter_count()
    print("Total Parameters: {:.2f}M".format(total_params / 1e6))

    with mpi.Session(MPI_LOCAL_RANK) as session:
        session.run(tf.initialize_all_variables())
        train_model(session=session, train_batches=train_batches,
                    validation_batches=validation_batches,
                    model_inputs=[inputs], train_step=train_step, loss=loss,
                    iterations=max_iterations)


if __name__ == "__main__":
    main()
