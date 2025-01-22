from pydantic import BaseModel, Field
import requests
import json

from model_configurations import get_model_configuration
from langchain_core.tools import StructuredTool



calendar_config = get_model_configuration('calendar')

class holidayRequest(BaseModel):
    country: str = Field(description="以2字母 iso code表示國家")
    year:int = Field(description="以yyyy表示年分")
    month:int = Field(description="以mm表示月份")

def getCalendar(country: str, year:int, month:int):
    url = 'https://calendarific.com/api/v2/holidays'
    parameters = {
        'api_key': calendar_config['api_key'],
        'country': country,
        'year': year,
        'month': month
    }
    response = requests.get(url, params=parameters)
    data = json.loads(response.text)
    return data

holiday_data = StructuredTool.from_function(
    func=getCalendar,
    name='Holiday_Calendar',
    description='特定年月份的節日有哪些',
    args_schema=holidayRequest
)