import openai
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from utils import count_n_tokens

MAX_TOKENS = 3800
COMPLETIION_TOKENS = 1024
key = "
openai.api_key = key

turbo_llm = ChatOpenAI(openai_api_key=key,
                       temperature=0.7,
                       model_name='gpt-3.5-turbo',
                       max_tokens=COMPLETIION_TOKENS
                       )

# System Prompts:
system_terms_prompt = SystemMessage(content="""you are a legal expert. you help users understand the terms and agreements. 
    you take as input a section from a terms and agreements contract. to help the user avoid potential risks and/or scams 
    your output is composed of 4 parts: 
    - Summary for the terms and agreements 
    - Potential risks and legal liabilities that might face the user
    - list of all  the legal penalties and their corresponding rules 
    - the do/don'ts for the user
    """)
system_summarize_prompt = SystemMessage(content="""

you are a legal expert. You make terms and agreements easy to understand for the user. For this, you take a list of 
potential risks, summaries, potential penalties and do's and don'ts highlighted in different sections of the 
documents. you compile all the information presented and you summarize it and then simplify it for the average person. 
Only use the most important 
information

your output is composed of 4 parts:
- Summary of the whole document
-  list of Potential risks and legal liabilities
- list of legal penalties and corresponding rules
- list of do's and don'ts


""")

term_prompt_len = count_n_tokens(system_terms_prompt.content)
summarize_prompt_len = count_n_tokens(system_summarize_prompt.content)


def llm_terms(text):
    """
    Extract potential risk and summarize each section
    :param text:
    :return:
    """

    human_terms_prompt = HumanMessagePromptTemplate.from_template("{term}")
    chat_prompt = ChatPromptTemplate.from_messages([system_terms_prompt, human_terms_prompt])
    result = turbo_llm(chat_prompt.format_prompt(term=text).to_messages()).content
    return result


def summarize_results(text):
    """
    Summarizes the result from the terms extraction
    :param text:
    :return:
    """

    human_prompt = HumanMessagePromptTemplate.from_template("input: \n{text}\noutput:")
    chat_prompt = ChatPromptTemplate.from_messages([system_summarize_prompt, human_prompt])
    result = turbo_llm(chat_prompt.format_prompt(text=text).to_messages()).content

    return result


