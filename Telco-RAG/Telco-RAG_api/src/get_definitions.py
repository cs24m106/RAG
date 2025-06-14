import os
from docx import Document
vocab_file = os.path.join(os.path.dirname(__file__), 'resources', '3GPP_vocabulary.docx')

def read_docx(file_path):
    """Reads a .docx file and categorizes its content into terms and abbreviations."""
    doc = Document(file_path)
    
    processing_terms = False
    processing_abbreviations = False
    start = 0 
    terms_definitions = {}
    abbreviations_definitions = {}
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if "References" in text:
            start += 1
        if start >= 2:
            if "Terms and definitions" in text:
                processing_terms = True
                processing_abbreviations = False
                
            elif "Abbreviations" in text:
                processing_abbreviations = True
                processing_terms = False
            else:
                if processing_terms and ':' in text:
                    term, definition = text.split(':', 1)
                    terms_definitions[term.strip()] = definition.strip().rstrip('.')
                elif processing_abbreviations and '\t' in text:
                    abbreviation, definition = text.split('\t', 1)
                    if len(abbreviation) > 1:
                        abbreviations_definitions[abbreviation.strip()] = definition.strip()
                
    return terms_definitions, abbreviations_definitions

def preprocess(text, lowercase=True):
    """Converts text to lowercase and removes punctuation."""
    if lowercase:
        text = text.lower()
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for char in punctuations:
        text = text.replace(char, '')
    return text

def find_and_filter_terms(terms_dict, sentence):
    """Finds terms in the given sentence, case-insensitively, and filters out shorter overlapping terms."""
    lowercase_sentence = preprocess(sentence, lowercase=True)
    
    matched_terms = {term: terms_dict[term] for term in terms_dict if preprocess(term) in lowercase_sentence}
    
    final_terms = {}
    for term in matched_terms:
        if not any(term in other and term != other for other in matched_terms):
            final_terms[term] = matched_terms[term]
            
    return final_terms

def find_and_filter_abbreviations(abbreviations_dict, sentence):
    """Finds abbreviations in the given sentence, case-sensitively, and filters out shorter overlapping abbreviations."""
    processed_sentence = preprocess(sentence, lowercase=False)  
    words = processed_sentence.split() 
    
    matched_abbreviations = {word: abbreviations_dict[word] for word in words if word in abbreviations_dict}
    
    final_abbreviations = {}
    sorted_abbrs = sorted(matched_abbreviations, key=len, reverse=True)
    for abbr in sorted_abbrs:
        if not any(abbr in other and abbr != other for other in sorted_abbrs):
            final_abbreviations[abbr] = matched_abbreviations[abbr]
    
    return final_abbreviations

def find_terms_and_abbreviations_in_sentence(terms_dict, abbreviations_dict, sentence):
    """Finds and filters terms and abbreviations in the given sentence.
       Abbreviations are matched case-sensitively, terms case-insensitively, and longer terms are prioritized."""
    matched_terms = find_and_filter_terms(terms_dict, sentence)
    matched_abbreviations = find_and_filter_abbreviations(abbreviations_dict, sentence)

    formatted_terms = [f"{term}: {definition}" for term, definition in matched_terms.items()]
    formatted_abbreviations = [f"{abbr}: {definition}" for abbr, definition in matched_abbreviations.items()]

    return formatted_terms, formatted_abbreviations

def get_def(sentence):
    terms_definitions, abbreviations_definitions = read_docx(vocab_file)
    formatted_terms, formatted_abbreviations = find_terms_and_abbreviations_in_sentence(terms_definitions, abbreviations_definitions, sentence)
    defined = []
    for term in formatted_terms:
        defined.append(term[:3])
    for abbreviation in formatted_abbreviations:
        defined.append(abbreviation[:3])

def define_TA_question(sentence):
    terms_definitions, abbreviations_definitions = read_docx(vocab_file)
    formatted_terms, formatted_abbreviations = find_terms_and_abbreviations_in_sentence(terms_definitions, abbreviations_definitions, sentence)
    terms = '\n'.join(formatted_terms)
    abbreviations = '\n'.join(formatted_abbreviations)
    question = f"""{sentence}\n
Terms and Definitions:\n
{terms}\n

Abbreviations:\n
{abbreviations}\n
"""
    return question
