# ------------ setup environment ------------ 
import logging, preheader # import for custom logger
logger = logging.getLogger(__name__) # Setup logging

from postgreSuperset.analysis import execute_pg_query
from chromaDatabase.ollama_pdf import initialize_database, prepare_dataset, rag_pipeline, compare_str, submit_prompt

import json
# ------------ configurations ------------ 
from postgreSuperset.ollama_sql_ver2 import datebase
import pandas as pd
table_name = 'public.hrly_kpi_1' # using previously used hardcord attrib, to keep context of each column
# TO DO: if yaml file can be made with all information needed, instead of manual description, it can be automated

# ------------ functions ------------ 
# Helper function to extract SQL from the generated response
def refractor_response(generated):
    logger.info(f"Need to refractor (into a parsable dict form) the generated response: {generated}")
    output = {}
    output['sql_query'] = generated
    output['explanation'] = None

    # if the generated response follows the given format
    start = generated.find('{') 
    end = generated.find('}', start) + 1 # sqlcoder gives lots of unnecessary text, lets stick with first '{' & '}'
    logger.info(f"'{{' start-pos:{start}, '}}' end-pos:{end}")
    if start != -1 and end == 0:
        # '{' found but '}' not found, append '}'
        generated = generated + '}'
        end = len(generated)
    if start != -1 and end != -1:
        refractored = generated[start:end]    
        try:
            output = json.loads(refractored)
            if output == {}:
                raise ValueError(f"Ollama response is empty or invalid JSON. \n Response Generated: \n{generated}")
        except Exception as e:
            logger.error(f"Failed to parse Ollama response as JSON: {e}\nRefractored Response: \n{refractored}")
    
    if output and 'sql_query' in output: # return if 'sql_query' key is found in parsed dict
        return output
    
    # Try to extract SQL if sqlcoder returns code block formatting (e.g., ```sql ... ```)
    if generated:
        code_start = generated.find("```")
        if code_start != -1:
            # Try to find the next code block end
            code_end = generated.find("```", code_start + 3)
            if code_end != -1:
                # Extract the code block content
                sql_block = generated[code_start + 3:code_end].strip()
                # Remove 'sql' language tag if present
                if sql_block.lower().startswith("sql"):
                    sql_block = sql_block[3:].strip()
                explanation = (generated[:code_start] + generated[code_end + 3:]).strip()
                output = {
                    "sql_query": sql_block,
                    "explanation": explanation
                }
                return output
    
    return output


# Prompt Handler function that requests llm to generate response to the given question aggregating all inputs together
def fetch_sql_generation(question: str, table_dict: dict[str, list[str]]={}, context: list[str]=[]):
    prompt = f"""
### Instructions:
You are a data analyst using Apache Superset with a PostgreSQL database.
You are also an expert in Wireless LTE/4G technology.
Your task is to convert a question into a SQL query, given a Postgres database schema.
Adhere to these rules:
- **Deliberately go through the question and database schema word by word** to appropriately answer the question
- **Use Table Aliases** to prevent ambiguity (if the query is ambiguous, give the most relavent answer)
- **Use only the Table Name, the attributes, i.e. column names provived within "Input" section** to form the sql query, add the schmema to your memory to answer carefully
- **DO NOT use any of the references from "User's Query"** as it is without referencing it with "Input" section.
- Try to include the **top 3 correlated attributes** along with the given query to give a better understanding of the data pattern.
- Try to **avoid using null values for comparison** based queries, unless explicitly asked
- **Follow the given Response Format strictly**, no additional texts or comments or examples outside the format is entertained
- Just return the response in **plain text** with no markdown based formatting


### Input:
Database schema:
{'\n'.join([f"CREATE TABLE {table} (\n{'\n'.join([str(attr.get()) for attr in attributes])}\n);" for table, attributes in table_dict.items()])}

Retreived Context:
{'\n'.join(context)}
        
Given the user query, Generate a SQL query that answers the question based on the table schemas and retr provided to understand the analogy between the attributes.
User query: "{question}"


### Response Format:
{{
    "sql_query": "give the sql query as plain text without any formatting, and no comments in between it",
    "explanation": "Reasoning: A brief explanation of the SQL query and how it relates to the user query",
}}
    """
    generated = submit_prompt(prompt)
    return refractor_response(generated)


# PostgreSQL handler to execute generated sql with error handling
def run_sql_query(response):
    if not response or 'sql_query' not in response:
        logger.error("No SQL query found in the response.")
        return None
    sql_query = response['sql_query']
    if not isinstance(sql_query, str):
        logger.error("SQL query is not a string.")
        return None

    try:
        logger.info(f"Executing SQL query:\n{sql_query}")
        result = execute_pg_query(sql_query)
        if result is not None:
            logger.info(f"Query executed successfully!")
        else:
            logger.error("Query execution returned no result!")
        return result
    except Exception as e:
        logger.error(f"Error executing SQL query: {e}")
        return None



# main() -> conc only on retreival
if __name__ == "__main__":
    config = initialize_database()
    prepare_dataset()

    # Example usage: Define Question to query the RAG model
    questions = [
        "What are the worst 10 performing sites based on RRCDR? in last 24 hours, list in descending order.", # complex query, filtering 2 times
        "What are the best 10 performing sites based on RRCSR?", # attrib is expected to be found
        """
        Pick last 2hr data all 40K sites all KPIs in hourly_kpi table, choose rrc conn setup succ rate , rrcdr as target KPIs. 
            - drop the cell id, drop the time, for rrc.conn.setup.succ rate, create 10 bins: 0-10, 11-20, ...91-100.
            - pick the smallest bin and pick samples from all bins with the smallest bin value --> ensure cellid is not duplicated.
            - Make the hour as configurable so we can change it to 24* hr to filter multiple days
        """ # big procedure
    ]
    answers = [] # add answers to measure similarity

    for i in range(len(questions)):
        logger.debug("\n\n")
        logger.info(f">>> Question: {questions[i]}")
        pure_retrieval = (not config["rag_generate"])
        _, context = rag_pipeline(questions[i], retrieve_only=pure_retrieval, should_enhance=pure_retrieval) # change config
        response = fetch_sql_generation(questions[i], 
                                        table_dict={table_name: datebase},
                                        context=context)
        if response != None:
            logger.info("######### ------ Response from LLM Generated ------ #########\n%s", json.dumps(response, indent=4))
            if 0 <= i < len(answers):
                logger.info(f">>> Expected Answer [similarity score w.r.t llm response = {compare_str(answers[i], response['sql_query'])}]: {answers[i]}")
        
        data = run_sql_query(response)
        if data is not None:
            logger.debug(data.to_string(index=False))
        else:
            logger.warning("No data returned from query.")
        input("\nPAUSED! Press <ENTER> to continue...")
    