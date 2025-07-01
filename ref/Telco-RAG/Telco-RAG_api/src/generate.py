import re
import traceback
from src.LLMs.LLM import submit_prompt_flex
import logging # Setup logging
logger = logging.getLogger(__name__)

def generate(question, model_name):
    """Generate a response using the GPT model given a structured question object."""
    # Constructing the content context from the question object
    content = '\n'.join(question.context)
    prompt = f"""
Please answer the following question:
{question.query}

Considering the following context:
{content}

Please answer the following question, add between paranthesis the retrieval(e.g. Retrieval 3) that you used for each eleement of your reasoning:
{question.question}
    """

    try:    
        logger.info(f"Generated prompt for LLM completion: \n{prompt}")
        
        predicted_answers_str = submit_prompt_flex(prompt, model_name=model_name)
        logger.info(f"Model response generated successfully. Predicted Answer:\n{predicted_answers_str}")

        context = f"The retrieved context provided to the LLM is:\n{content}"
        return predicted_answers_str, context, question.question

    except Exception as e:
        # logger the error and returning a failure indicator
        logger.error(f"An error occurred: {e}")
        return None, None, None


def check_question(question, answer, options, model_name='gpt-4o-mini'):
    """
    This function checks if the answer provided for a non-JSON formatted question is correct. 
    It dynamically selects the model based on the model_name provided and constructs a prompt 
    for the AI model to generate an answer. It then compares the generated answer with the provided 
    answer to determine correctness.

    Parameters:
    - question: A dictionary containing the question, options, and context.
    - model_name: Optional; specifies the model to use. Defaults to 'mistralai/Mixtral-8x7B-Instruct-v0.1' 
    if not provided or if the default model is indicated.

    Returns:
    - A tuple containing the updated question dictionary and a boolean indicating correctness.
    """
    # Extracting options => Convert the options dict to a text format: "option {no}: value"
    options_text = '\n'.join([f"{key}: {value}" for key, value in options.items()])

    content = '\n'.join(question.context)

    syst_prompt = f"""
Please provide the answers to the following multiple choice question.
{question.query}

Considering the following context:
{content}

Please provide the answers to the following multiple choice question.
{question.question}

Options:
Write only the option number corresponding to the correct answer:\n{options_text}

Answer format should be: Answer option <option_id>
    """

    try:
        logger.info(f"Generated system prompt for LLM completion: \n{syst_prompt}")
        # Generating the model's response based on the constructed prompt.
        predicted_answers_str = submit_prompt_flex(syst_prompt, model_name=model_name)
        predicted_answers_str = predicted_answers_str.replace('"\n', '",\n')

        # Finding and comparing the predicted answer to the actual answer.
        pred_ans_id = find_option_number(predicted_answers_str)
        act_ans_id = find_option_number(answer)
        logger.info(f"Model response generated successfully. \nActual answer[opt-id:{act_ans_id}]: {answer} \nPredicted Answer [opt-id:{pred_ans_id}]: {predicted_answers_str}")

        if act_ans_id == pred_ans_id:
            logger.info("Answers match => Correct!\n")
        else:
            logger.info("Answers dont match => Wrong!\n")
        return  act_ans_id == pred_ans_id, f"Option {pred_ans_id}", syst_prompt
    
    except Exception as e:
        # Error handling to catch and report errors more effectively.
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
        return None, False

def find_option_number(text):
    """
    Finds all the occurrences of numbers preceded by the word 'option' in a given text.

    Parameters:
    - text: The text to search for 'option' followed by numbers.

    Returns:
    - A list of strings, each representing a number found after 'option'. The numbers are returned as strings.
    - If no matches are found, an empty list is returned.
    """
    try:
        text =  text.lower()
        # Define a regular expression pattern to find 'option' followed by non-digit characters (\D*), 
        # and then one or more digits (\d+)
        pattern = r'option\D*(\d+)'
        # Find all matches of the pattern in the text
        matches = re.findall(pattern, text)
        return matches  # Return the list of found numbers as strings
    except Exception as e:
        print(f"An error occurred while trying to find option numbers in the text: {e}")
        return []