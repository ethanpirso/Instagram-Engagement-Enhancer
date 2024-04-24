from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel

def perform_lda(labels_list, num_topics=10):
    """Performs LDA topic modeling."""
    dictionary = corpora.Dictionary(labels_list)
    corpus = [dictionary.doc2bow(text) for text in labels_list]

    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100, update_every=1, passes=10, alpha='auto', per_word_topics=True)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=labels_list, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model_lda.get_coherence()

    return lda_model, dictionary, corpus, coherence

def find_optimal_topics(labels_list, start=2, limit=20, step=3):
    """Determines the optimal number of topics for LDA."""
    coherence_values = []
    model_list = []
    topic_numbers = []  # List to track the number of topics

    for num_topics in range(start, limit, step):
        model, _, _, coherence = perform_lda(labels_list, num_topics=num_topics)
        model_list.append(model)
        coherence_values.append(coherence)
        topic_numbers.append(num_topics)  # Track the number of topics

    # Find the index of the highest coherence value
    max_coherence_idx = coherence_values.index(max(coherence_values))
    optimal_num_topics = topic_numbers[max_coherence_idx]  # Get the optimal number of topics

    return model_list, coherence_values, optimal_num_topics  # Return the optimal number of topics as well
