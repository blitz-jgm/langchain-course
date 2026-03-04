from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Scheme for a source used by the agent"""
    
    url:str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Scheme for the response of the agent with answer and sources"""
    
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="The list ofsources used to generate the answer")




llm = ChatAnthropic(model="claude-haiku-4-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")]})
    print(result)


if __name__ == "__main__":
    main()
