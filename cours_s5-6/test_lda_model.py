# /bin/env python
# -*- coding: utf-8 -*-

"""how to annotate videos sub-corpus with gensim lda model
"""

import topic_modeling_lda as lda_model

# gensim config file
conf_path = "./gensim.json"

# preprocess corpus
corpus, docs, id2word, dictionary = lda_model.preprocess(conf_path)

# print(dictionary)
# print(docs)
# print(id2word)
# print("unique lemmas:", len(dictionary))
# print("number of documents:", len(docs))
# print("number of tokens in each documents:", ", ".join([str(len(d)) for d in docs]))

# train model
model = lda_model.train_model(conf_path, corpus, docs, id2word, dictionary)

# save model
lda_model.save_model(conf_path, model, corpus, dictionary)

# get model infos by loading model
lda_model.write_infos(conf_path)

# add annots to the corpus by loading model
lda_model.write_annots(conf_path, 'gensim_topics')

