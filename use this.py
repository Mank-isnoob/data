import codecs
import os
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

def fetch_data(cand, ref):
    """Read candidate and reference sentences from files."""
    with codecs.open(ref, 'r', 'utf-8') as reference_file:
        references = reference_file.readlines()
    
    with codecs.open(cand, 'r', 'utf-8') as candidate_file:
        candidate = candidate_file.readlines()
    
    return candidate, references

def preprocess_sentences(candidate, references):
    """Preprocess the sentences by splitting into words."""
    preprocessed_references = [[sentence.strip().split()] for sentence in references]
    preprocessed_candidate = [sentence.strip().split() for sentence in candidate]

    return preprocessed_candidate, preprocessed_references

def calculate_sentence_bleu(candidate, references):
    """Calculate BLEU score for each individual sentence."""
    chencherry = SmoothingFunction()
    individual_scores = []
    
    for cand, ref in zip(candidate, references):
        score = sentence_bleu(ref, cand, smoothing_function=chencherry.method1)
        individual_scores.append(score)
    
    return individual_scores

def calculate_corpus_bleu(candidate, references):
    """Calculate the BLEU score for the entire corpus."""
    chencherry = SmoothingFunction()
    bleu_score = corpus_bleu(references, candidate, smoothing_function=chencherry.method1)
    return bleu_score

if __name__ == "__main__":
    # Define file paths here
    candidate_file = 'candidate.txt'
    reference_file = 'references.txt'
    
    # Fetch data from the defined file paths
    candidate, references = fetch_data(candidate_file, reference_file)
    
    # Preprocess sentences
    preprocessed_candidate, preprocessed_references = preprocess_sentences(candidate, references)
    
    # Calculate individual sentence BLEU scores
    sentence_bleu_scores = calculate_sentence_bleu(preprocessed_candidate, preprocessed_references)
    
    # Calculate corpus BLEU score
    corpus_bleu_score = calculate_corpus_bleu(preprocessed_candidate, preprocessed_references)
    
    # Write individual sentence BLEU scores and corpus BLEU score to output file
    with open('bleu_scores.txt', 'w', encoding='utf-8') as out_file:
        out_file.write("Individual Sentence BLEU Scores:\n")
        for idx, score in enumerate(sentence_bleu_scores, 1):
            out_file.write(f"Sentence {idx}: {score}\n")
        
        out_file.write("\nCorpus BLEU Score:\n")
        out_file.write(str(corpus_bleu_score))
