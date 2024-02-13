# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder
from tensorflow.compat.v1.contrib.tensorboard.plugins import projector

class SummarizationModel(tf.Module):
    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def _add_placeholders(self):
        hps = self._hps

        # encoder part
        self._enc_batch = tf.Variable(initial_value=tf.zeros([hps.batch_size, None], dtype=tf.int32), trainable=False, name='enc_batch')
        self._enc_lens = tf.Variable(initial_value=tf.zeros([hps.batch_size], dtype=tf.int32), trainable=False, name='enc_lens')
        self._enc_padding_mask = tf.Variable(initial_value=tf.zeros([hps.batch_size, None], dtype=tf.float32), trainable=False, name='enc_padding_mask')
        if FLAGS.pointer_gen:
            self._enc_batch_extend_vocab = tf.Variable(initial_value=tf.zeros([hps.batch_size, None], dtype=tf.int32), trainable=False, name='enc_batch_extend_vocab')
            self._max_art_oovs = tf.Variable(initial_value=tf.zeros([], dtype=tf.int32), trainable=False, name='max_art_oovs')

        # decoder part
        self._dec_batch = tf.Variable(initial_value=tf.zeros([hps.batch_size, hps.max_dec_steps], dtype=tf.int32), trainable=False, name='dec_batch')
        self._target_batch = tf.Variable(initial_value=tf.zeros([hps.batch_size, hps.max_dec_steps], dtype=tf.int32), trainable=False, name='target_batch')
        self._dec_padding_mask = tf.Variable(initial_value=tf.zeros([hps.batch_size, hps.max_dec_steps], dtype=tf.float32), trainable=False, name='dec_padding_mask')

        if hps.mode == "decode" and hps.coverage:
            self.prev_coverage = tf.Variable(initial_value=tf.zeros([hps.batch_size, None], dtype=tf.float32), trainable=False, name='prev_coverage')

    def _make_feed_dict(self, batch, just_enc=False):
        feed_dict = {}
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
        if FLAGS.pointer_gen:
            feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed_dict[self._max_art_oovs] = batch.max_art_oovs
        if not just_enc:
            feed_dict[self._dec_batch] = batch.dec_batch
            feed_dict[self._target_batch] = batch.target_batch
            feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
        return feed_dict

    def _add_encoder(self, encoder_inputs, seq_len):
        with tf.compat.v1.variable_scope('encoder'):
            cell_fw = tf.keras.layers.LSTMCell(self._hps.hidden_dim, kernel_initializer=tf.keras.initializers.RandomUniform(), state_is_tuple=True)
            cell_bw = tf.keras.layers.LSTMCell(self._hps.hidden_dim, kernel_initializer=tf.keras.initializers.RandomUniform(), state_is_tuple=True)
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)  # concatenate the forwards and backwards states
        return encoder_outputs, fw_st, bw_st

    def _reduce_states(self, fw_st, bw_st):
        hidden_dim = self._hps.hidden_dim
        with tf.compat.v1.variable_scope('reduce_final_st'):
            w_reduce_c = tf.compat.v1.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
            w_reduce_h = tf.compat.v1.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
            bias_reduce_c = tf.compat.v1.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
            bias_reduce_h = tf.compat.v1.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=tf.truncated_normal_initializer())

            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)
            return tf.compat.v1.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

    def _add_decoder(self, inputs):
        hps = self._hps
        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(hps.hidden_dim, state_is_tuple=True, kernel_initializer=tf.keras.initializers.RandomUniform())

        prev_coverage = self.prev_coverage if hps.mode == "decode" and hps.coverage else None

        outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(inputs, self._dec_in_state, self._enc_states,
                                                                              self._enc_padding_mask,
                                                                              cell, initial_state_attention=(hps.mode == "decode"),
                                                                              pointer_gen=hps.pointer_gen,
                                                                              use_coverage=hps.coverage,
                                                                              prev_coverage=prev_coverage)

        return outputs, out_state, attn_dists, p_gens, coverage

    def _calc_final_dist(self, vocab_dists, attn_dists):
        with tf.compat.v1.variable_scope('final_distribution'):
            vocab_dists = [p_gen * dist for (p_gen, dist) in zip(self.p_gens, vocab_dists)]
            attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(self.p_gens, attn_dists)]

            extended_vsize = self._vocab.size() + self._max_art_oovs
            extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
            vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

            batch_nums = tf.range(0, limit=self._hps.batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)
            attn_len = tf.shape(self._enc_batch_extend_vocab)[1]
            batch_nums = tf.tile(batch_nums, [1, attn_len])
            indices = tf.stack((batch_nums, self._enc_batch_extend_vocab), axis=2)
            shape = [self._hps.batch_size, extended_vsize]
            attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]

            final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

            return final_dists

    def _add_emb_vis(self, embedding_var):
        train_dir = os.path.join(self._hps.log_root, "train")
        vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
        with open(vocab_metadata_path, "w") as f:
            for i in range(self._vocab.size()):
                f.write(self._vocab.id2word(i) + '\n')

        vocab_emb = tf.Variable(embedding_var.read_value(), name='vocab_emb')
        self.embedding_placeholder = tf.compat.v1.placeholder(tf.float32, [self._vocab.size(), self._hps.emb_dim])
        self.embedding_init = vocab_emb.assign(self.embedding_placeholder)
        summary_writer = tf.compat.v1.summary.FileWriter(train_dir)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = vocab_emb.name
        embedding.metadata_path = vocab_metadata_path
        projector.visualize_embeddings(summary_writer, config)

    def _add_seq2seq(self):
        hps = self._hps
        vsize = self._vocab.size()

        with tf.compat.v1.variable_scope('seq2seq'):
            # Some initializers
            self.rand_unif_init = tf.random.uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
            self.trunc_norm_init = tf.compat.v1.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

            # Add embedding matrix (shared by the encoder and decoder inputs)
            with tf.compat.v1.variable_scope('embedding'):
                if hps.fix_embedding:
                    embedding = tf.constant(self._vocab.embeddings, dtype=tf.float32)
                else:
                    embedding = tf.compat.v1.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32,
                                                  initializer=self.rand_unif_init)
                self._emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)  # tensor with shape (batch_size, max_enc_steps, emb_size)
                self._emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)]  # list length max_dec_steps containing shape (batch_size, emb_size)

            # Add the encoder.
            enc_outputs, fw_st, bw_st = self._add_encoder(self._emb_enc_inputs, self._enc_lens)

            # Track encoder states for attention calculation
            self._enc_states = enc_outputs

            # Reduce the final encoder state to the right size for the decoder
            self._dec_in_state = self._reduce_states(fw_st, bw_st)

            # Add the decoder.
            with tf.compat.v1.variable_scope('decoder'):
                decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = self._add_decoder(self._emb_dec_inputs)

            # Add the output projection to obtain the vocabulary distribution
            with tf.compat.v1.variable_scope('output_projection'):
                w = tf.compat.v1.get_variable('w', [hps.hidden_dim, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
                v = tf.compat.v1.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
                vocab_scores = []  # store the vocab distribution here; allow for concatenation of sub-batch vocab distributions
                for i, output in enumerate(decoder_outputs):
                    if i > 0:
                        tf.compat.v1.get_variable_scope().reuse_variables()
                    vocab_scores.append(tf.nn.xw_plus_b(output, w, v))  # apply the linear layer

                # Concatenate the scores from each decoder output tensor into a tensor of shape (batch_size, max_dec_steps, vsize)
                vocab_dists = tf.transpose(tf.stack(vocab_scores), [1, 0, 2])

            # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
            if hps.pointer_gen:
                final_dists = self._calc_final_dist(vocab_dists, self.attn_dists)
            else:  # final distribution is just vocabulary distribution
                final_dists = vocab_dists

            if hps.mode in ['train', 'eval']:
                # Calculate the loss
                with tf.compat.v1.variable_scope('loss'):
                    if hps.pointer_gen:
                        # Calculate the loss per step
                        # This is fiddly; we use tf.gather_nd to pick out the probabilities of the actual words
                        loss_per_step = []  # will be list length max_dec_steps containing shape (batch_size)
                        batch_nums = tf.range(0, limit=hps.batch_size)  # shape (batch_size)
                        for dec_step, dist in enumerate(final_dists):
                            targets = self._target_batch[:, dec_step]  # The indices of the target words. shape (batch_size)
                            indices = tf.stack((batch_nums, targets), axis=1)  # shape (batch_size, 2)
                            gold_probs = tf.gather_nd(dist, indices)  # shape (batch_size). prob of correct words on this step
                            losses = -tf.math.log(gold_probs)
                            loss_per_step.append(losses)

                        # Apply dec_padding_mask and get loss
                        self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)

                        # Calculate coverage loss from the attention distributions
                        if hps.coverage:
                            with tf.compat.v1.variable_scope('coverage_loss'):
                                self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
                                self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
                    else:  # baseline model
                        self._loss = tf.compat.v1.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self._target_batch, self._dec_padding_mask, average_across_timesteps=True, average_across_batch=True)

                # Add train op
                if hps.optimizer == 'adam':
                    optimizer = tf.compat.v1.train.AdamOptimizer(hps.lr)
                elif hps.optimizer == 'adagrad':
                    optimizer = tf.compat.v1.train.AdagradOptimizer(hps.lr, initial_accumulator_value=hps.adagrad_init_acc)
                elif hps.optimizer == 'sgd':
                    optimizer = tf.compat.v1.train.GradientDescentOptimizer(hps.lr)
                elif hps.optimizer == 'rmsprop':
                    optimizer = tf.compat.v1.train.RMSPropOptimizer(hps.lr)
                else:
                    raise ValueError("Unknown optimizer type %s" % hps.optimizer)
                self._train_op = optimizer.minimize(self._loss, global_step=self.global_step)

                # Summary op
                tf.compat.v1.summary.scalar('loss', self._loss)
                if hps.pointer_gen and hps.coverage:
                    tf.compat.v1.summary.scalar('coverage_loss', self._coverage_loss)
                    tf.compat.v1.summary.scalar('total_loss', self._total_loss)
            elif hps.mode == "decode":
                # We run decode beam search mode one decoder step at a time
                assert len(final_dists) == 1  # final_dists is a singleton list containing shape (batch_size, extended_vsize)
                final_dists = final_dists[0]
                top_k_probs, self._top_k_ids = tf.nn.top_k(final_dists, hps.batch_size * 2)  # take the k largest probs. note batch_size=beam_size in decode mode
                self._top_k_log_probs = tf.math.log(top_k_probs)

        # Add the train_op to the graph
        if hps.mode == "train":
            return self._train_op
        elif hps.mode == "decode":
            return self._top_k_ids

    def build_graph(self):
        start_time = time.time()
        tf.compat.v1.logging.info('Building graph...')
        self._add_placeholders()
        self._add_seq2seq()
        self._add_emb_vis(self._emb_dec_inputs)
        tf.compat.v1.logging.info('Time to build graph: %s seconds', time.time() - start_time)

