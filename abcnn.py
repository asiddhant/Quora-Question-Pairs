from __future__ import print_function
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Convolution1D, Convolution2D, AveragePooling1D, GlobalAveragePooling1D, ZeroPadding1D, ZeroPadding2D
from keras.layers import Dense, Lambda, merge, TimeDistributed, RepeatVector, Permute,  Reshape, Dropout, BatchNormalization
import numpy as np
import pandas as pd

def compute_cosine_match_score(l_r):
    l, r = l_r
    return K.batch_dot( K.l2_normalize(l, axis=-1), K.l2_normalize(r, axis=-1), axes=[2, 2])


def compute_euclidean_match_score(l_r):
    l, r = l_r
    denominator = 1.+K.sqrt(-2 * K.batch_dot(l, r, axes=[2, 2]) + 
                            K.expand_dims(K.sum(K.square(l), axis=2), 2) + 
                            K.expand_dims(K.sum(K.square(r), axis=2), 1))
    denominator = K.maximum(denominator, K.epsilon())
    return 1. / denominator

def MatchScore(l, r, mode="euclidean"):
    if mode == "euclidean":
        return merge([l, r],mode=compute_euclidean_match_score,output_shape=lambda shapes: (None, shapes[0][1], shapes[1][1]))
    if mode == "cos":
        return merge([l, r],mode=compute_cosine_match_score,output_shape=lambda shapes: (None, shapes[0][1], shapes[1][1]))


def ABCNN(left_seq_len, right_seq_len, embed_dimensions, nb_filter, filter_widths, depth=3, dropout=0.5, 
          abcnn_1=True, abcnn_2=True, collect_sentence_representations=False, mode="euclidean"):

    assert depth >= 1, "Need at least one layer to build ABCNN"
    assert not (depth == 1 and abcnn_2), "Cannot build ABCNN-2 with only one layer!"
    if type(filter_widths) == int:
        filter_widths = [filter_widths] * depth
    assert len(filter_widths) == depth

    print("Using %s match score" % mode)

    left_sentence_representations = []
    right_sentence_representations = []

    left_input = Input(shape=(left_seq_len, embed_dimensions))
    right_input = Input(shape=(right_seq_len, embed_dimensions))

    left_embed = left_input
    right_embed = right_input

    filter_width = filter_widths.pop(0)

    if abcnn_1:
        
        match_score = MatchScore(left_embed, right_embed, mode=mode)

        # Compute Attention
        attention_left = TimeDistributed(
            Dense(embed_dimensions, activation="relu"), input_shape=(left_seq_len, right_seq_len))(match_score)
        match_score_t = Permute((2, 1))(match_score)
        attention_right = TimeDistributed(
            Dense(embed_dimensions, activation="relu"), input_shape=(right_seq_len, left_seq_len))(match_score_t)

        left_reshape = Reshape((1, attention_left._keras_shape[1], attention_left._keras_shape[2]))
        right_reshape = Reshape((1, attention_right._keras_shape[1], attention_right._keras_shape[2]))

        attention_left = left_reshape(attention_left)
        left_embed = left_reshape(left_embed)

        attention_right = right_reshape(attention_right)
        right_embed = right_reshape(right_embed)

        # Concat Attention
        left_embed = merge([left_embed, attention_left], mode="concat", concat_axis=1)
        right_embed = merge([right_embed, attention_right], mode="concat", concat_axis=1)

        # Padding for wider Convolution
        left_embed_padded = ZeroPadding2D((filter_width - 1, 0))(left_embed)
        right_embed_padded = ZeroPadding2D((filter_width - 1, 0))(right_embed)

        # 2D convolutions to take care of channels if present. We are still doing 1-D convolutions.
        conv_left = Convolution2D(
            nb_filter=nb_filter, nb_row=filter_width, nb_col=embed_dimensions, activation="tanh", border_mode="valid",
            dim_ordering="th")(left_embed_padded)

        # Reshape and Permute to get back to 1-D
        conv_left = (Reshape((conv_left._keras_shape[1], conv_left._keras_shape[2])))(conv_left)
        conv_left = Permute((2, 1))(conv_left)

        conv_right = Convolution2D(
            nb_filter=nb_filter, nb_row=filter_width, nb_col=embed_dimensions, activation="tanh",border_mode="valid",
            dim_ordering="th")(right_embed_padded)

        # Reshape and Permute to get back to 1-D
        conv_right = (Reshape((conv_right._keras_shape[1], conv_right._keras_shape[2])))(conv_right)
        conv_right = Permute((2, 1))(conv_right)

    # Without Attention
    else:
        left_embed_padded = ZeroPadding1D(filter_width - 1)(left_embed)
        right_embed_padded = ZeroPadding1D(filter_width - 1)(right_embed)
        conv_left = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(left_embed_padded)
        conv_right = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(right_embed_padded)

    conv_left = Dropout(dropout)(conv_left)
    conv_right = Dropout(dropout)(conv_right)

    pool_left = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_left)
    pool_right = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_right)

    if collect_sentence_representations or depth == 1:  # always collect last layers global representation
        left_sentence_representations.append(GlobalAveragePooling1D()(conv_left))
        right_sentence_representations.append(GlobalAveragePooling1D()(conv_right))

    for i in range(depth - 1):
        
        filter_width = filter_widths.pop(0)
        
        pool_left = ZeroPadding1D(filter_width - 1)(pool_left)
        pool_right = ZeroPadding1D(filter_width - 1)(pool_right)
        
        # Wide Convolution
        conv_left = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(pool_left)
        conv_right = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(pool_right)

        if abcnn_2:
            conv_match_score = MatchScore(conv_left, conv_right, mode=mode)

            # Compute Attention
            conv_attention_left = Lambda(lambda match: K.sum(match, axis=-1), output_shape=(conv_match_score._keras_shape[1],))(conv_match_score)
            conv_attention_right = Lambda(lambda match: K.sum(match, axis=-2), output_shape=(conv_match_score._keras_shape[2],))(conv_match_score)

            conv_attention_left = Permute((2, 1))(RepeatVector(nb_filter)(conv_attention_left))
            conv_attention_right = Permute((2, 1))(RepeatVector(nb_filter)(conv_attention_right))

            conv_left = merge([conv_left, conv_attention_left], mode="mul")
            conv_right = merge([conv_right, conv_attention_right], mode="mul")

        conv_left = Dropout(dropout)(conv_left)
        conv_right = Dropout(dropout)(conv_right)

        pool_left = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_left)
        pool_right = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_right)

        if collect_sentence_representations or (i == (depth - 2)):
            left_sentence_representations.append(GlobalAveragePooling1D()(conv_left))
            right_sentence_representations.append(GlobalAveragePooling1D()(conv_right))

    # Merge Collected Sentence Representations
    left_sentence_rep = left_sentence_representations.pop(-1)
    if left_sentence_representations:
        left_sentence_rep = merge([left_sentence_rep] + left_sentence_representations, mode="concat")

    right_sentence_rep = right_sentence_representations.pop(-1)
    if right_sentence_representations:
        right_sentence_rep = merge([right_sentence_rep] + right_sentence_representations, mode="concat")

    global_representation = merge([left_sentence_rep, right_sentence_rep], mode="concat")
    global_representation = Dropout(dropout)(global_representation)

    # Add Fully Connected Layer on Top
    finalout = Dense(1, activation="sigmoid")(global_representation)

    return Model([left_input, right_input], output=finalout)


def _main():

    num_samples = 2332

    left_seq_len = 10
    right_seq_len = 10

    embed_dimensions = 100

    full_data=pd.read_csv('full_data.csv',header=None)
    full_data=np.array(full_data)

    X=[np.reshape(full_data[:,:left_seq_len*embed_dimensions],(num_samples,left_seq_len,embed_dimensions)),
       np.reshape(full_data[:,left_seq_len*embed_dimensions:(left_seq_len+right_seq_len)*embed_dimensions],
                  (num_samples,right_seq_len,embed_dimensions))]

    Y=full_data[:,(left_seq_len+right_seq_len)*embed_dimensions]

    nb_filter = 100
    filter_width = [3,3,3]

    model = ABCNN(left_seq_len=left_seq_len, right_seq_len=right_seq_len, depth=3, embed_dimensions=embed_dimensions, nb_filter=nb_filter, 
                  filter_widths=filter_width, collect_sentence_representations=False, abcnn_1=True, abcnn_2=True, mode="euclidean")

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    model.fit(X, Y, nb_epoch=100 ,batch_size=128, validation_split=0.2)
    print(model.predict(X)[0])

if __name__ == "__main__":
    _main()