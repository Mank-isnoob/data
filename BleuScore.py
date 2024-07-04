import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def read_references_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    reference_sets = content.split('\n\n')
    reference_sentences = [ref_set.split('\n') for ref_set in reference_sets]
    return reference_sentences

def read_predictions_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences]

def calculate_bleu_score(reference_sentences, predicted_sentences):
    assert len(reference_sentences) == len(predicted_sentences), "Number of reference sets and predicted sentences should be the same"

    bleu_scores = []
    chencherry = SmoothingFunction()

    for references, prediction in zip(reference_sentences, predicted_sentences):
        reference_list = [ref.split() for ref in references]  # Convert each reference to list of words
        prediction_list = prediction.split()
        score = sentence_bleu(reference_list, prediction_list, smoothing_function=chencherry.method1)
        bleu_scores.append(score)

    return bleu_scores

def write_bleu_scores_to_file(file_path, bleu_scores):
    with open(file_path, 'w', encoding='utf-8') as file:
        for i, score in enumerate(bleu_scores):
            file.write(f"BLEU score for sentence pair {i+1}: {score:.4f}\n")

# File paths
reference_file_path = 'references.txt'
prediction_file_path = 'predictions.txt'
output_file_path = 'bleu_scores.txt'

# Read sentences from files
reference_sentences = read_references_from_file(reference_file_path)
predicted_sentences = read_predictions_from_file(prediction_file_path)

# Calculate BLEU scores
bleu_scores = calculate_bleu_score(reference_sentences, predicted_sentences)

# Write BLEU scores to file
write_bleu_scores_to_file(output_file_path, bleu_scores)
