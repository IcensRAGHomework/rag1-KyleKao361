import base64
import json
import traceback

from pydantic import BaseModel, Field
from typing import List

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


from langchain.agents import AgentExecutor, create_openai_functions_agent, create_openai_tools_agent
from tools import holiday_data
from rich import print as pprint

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)
calendar_config = get_model_configuration('calendar')

store = {}

llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
)

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

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

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
    
    chain = final_prompt | llm
    response = chain.invoke({"input": question})
    
    return response.content
    
def generate_hw02(question):
    
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a restful api for calendar"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
            ("human", "以{few_shot_prompt}為json範例格式回答, 不要有多餘json文字")
        ]
    )

    tools = [holiday_data]
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=final_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    # agent_executor.return_intermediate_steps=True

    return agent_executor.invoke({'input': question, 'few_shot_prompt': few_shot_prompt})['output']
    
def generate_hw03(question2, question3):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a restful api for calendar"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
        ("human", "若是詢問節日清單以{few_shot_prompt}為json範例格式回答"
          "不要有多餘json文字,若是詢問是否要新增節日以{third_json_spec}為json範例格式回答, 不要有多餘json文字")
    ])

    third_json_spec = '{"Result":{"add": <是否該將節日加到前面回答的清單, 若已存在則為false反之則為true>,"reason": "<加入或不加入的理由以及該月份清單的所有節日名稱並說是幾月>"} }'
    prompt = prompt.partial(few_shot_prompt = few_shot_prompt, third_json_spec= third_json_spec)

    tools = [holiday_data]
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executer = AgentExecutor(agent=agent, tools=tools,verbose=False)

    history_handler = RunnableWithMessageHistory(
        agent_executer,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    result_part = history_handler.invoke(
        {"input": question2},
        config={"configurable": {"session_id": "holidays"}}
    )

    result_part = history_handler.invoke(
        {'input': question3},
        config={"configurable": {"session_id": "holidays"}}
    )

    return result_part['output']
    
def generate_hw04(question):
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', 'you are a ocr system to get score board and return score of input team with json like {fourth_json_spec}'),
        ('human', '{input}')
    ])
    fourth_json_spec = '{"Result":{"score": <score>} }'
    prompt_template = prompt_template.partial(fourth_json_spec = fourth_json_spec)

    messages = prompt_template.format_prompt(input=question).to_messages()
    messages.append(HumanMessage([{ 
        'type': 'image_url', 
        'image_url': {'url': f'data:image/jpeg;base64,{get_image("./baseball.png")}'}
    }]))
    response = llm.invoke(messages)
    return response.content
    
def get_image(path):
    with open(path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
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

print(generate_hw01('2024年台灣10月紀念日有哪些?'))
# print(generate_hw02('2024年台灣10月紀念日有哪些?'))
# print(generate_hw03('2024年台灣3月紀念日有哪些?', '根據先前的節日清單，這個節日{"date": "3-29", "name": "青年節"}是否有在該月份清單？'))
print(generate_hw04('請問中華台北的積分是多少'))