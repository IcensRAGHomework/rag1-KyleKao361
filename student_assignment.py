import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate,MessagesPlaceholder

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent, create_openai_tools_agent
from tools import holiday_data
from langchain import hub
from rich import print as pprint

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)
calendar_config = get_model_configuration('calendar')


def generate_hw01(question):
    examples= [
        {"input": "2024年台灣10月紀念日有哪些?", "output": '{"Result": [{"date": "2024-10-10","name": "國慶日"},{"date": "2024-10-25","name": "光復節"}]}'},
        {"input": "2024年台灣3月紀念日有哪些?", "output": '{"Result": [{"date": "2024-03-29","name": "青年節"}]}'}
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a restful api for calendar"),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    chain = final_prompt | llm
    response = chain.invoke({"input": question})
    
    return response.content
    
def generate_hw02(question):
    examples= [
        {"input": '[{"name":"NationalDay","description":"NationalDayisanationalholidayinTaiwan","country":{"id":"tw","name":"Taiwan"},"date":{"iso":"2024-10-10","datetime":{"year":2024,"month":10,"day":10}},"type":["Nationalholiday"],"primary_type":"Nationalholiday","canonical_url":"https://calendarific.com/holiday/taiwan/national-day","urlid":"taiwan/national-day","locations":"All","states":"All"},{"name":"Taiwan"sRetrocessionDay","description":"Taiwan"sRetrocessionDayisaobservanceinTaiwan","country":{"id":"tw","name":"Taiwan"},"date":{"iso":"2024-10-25","datetime":{"year":2024,"month":10,"day":25}},"type":["Observance"],"primary_type":"Observance","canonical_url":"https://calendarific.com/holiday/taiwan/taiwan-retrocession-day","urlid":"taiwan/taiwan-retrocession-day","locations":"All","states":"All"}]', "output": '{"Result": [{"date": "2024-10-10","name": "國慶日"},{"date": "2024-10-25","name": "光復節"}]}'}
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a restful api for calendar"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
            ("human", "以{few_shot_prompt}為json範例格式回答, 不要有多餘json文字")
        ]
    )

    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature'],
            cache= True
    )
    tools = [holiday_data]
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=final_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    # agent_executor.return_intermediate_steps=True


    return agent_executor.invoke({'input': question, 'few_shot_prompt': few_shot_prompt})['output']
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response

# pprint(generate_hw02('2024年台灣10月紀念日有哪些?'))