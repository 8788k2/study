# LLM + ROS Mobile Robot Automated Control
언어 모델은 그 자체로는 텍스트를 출력할 뿐 행동을 취할 수 없음. 

**에이전트는 고수준 작업을 수행하고 LLM을 추론 엔진으로 사용하여 어떤 행동을 취할지 결정하고 툴을 이용해 해당 행동을 실행하는 시스템**

에이전트 구성에 또 다른 에이전트 이용 가능

첫번째 목표 - langchain을 통해 에이전트 구성 (간단한 예제 + 유튜브 영상 활용) 

궁극적 목표 - 구체적으로 원하는 테스크(모바일 로봇 제어 관련) 정하고 langchain 활용하여 해당 테스크 수행하는 에이전트 구축
## 1. LangChain이란?
**llm을 보다 강력하게 활용하도록 도와주는 프레임 워크**로서 llm을 다양한 데이터 소스, 기능, 도구들과 결합시킬 수 있는 환경을 제공하여 지능적인 에이전트를 구축할 수 있게 해줌

구조

|기능|설명|활용 예제|
|---|---|-------|
|LLM|GPT-4, Claude, Mistral 등 LLM과 통합|OpenAI API를 활용한 자연어 처리|
|Chains|여러 개의 프롬프트와 LLM 호출을 연결|	사용자가 질문하면 LLM이 ROS 액션을 수행|
|Agents|	LLM이 자동으로 어떤 도구(API/모델 등)를 사용할지 결정|	LLM이 상황에 맞게 A* 경로 탐색, PPO 강화학습 등을 선택|
Tools|	에이전트가 호출할 수 있는 함수들의 집합|	ROS 서비스 호출, 데이터베이스 검색 등|
Memory|	LLM이 과거 정보를 기억하고 맥락을 유지|	자율주행 기록을 저장하고, 반복된 패턴을 학습|

랭체인을 활용하면 llm은 "의사결정모듈"로써 상황 이해 및 전략 수립등에 활용하고 디테일한 제어는 보다 적합한 툴을 활용할 수 있음


## 2. RAG란 (Retrieval Augmented Generation)?
LLM이 정보를 검색하여 활용할 수 있도록 하여 더욱 정확하고 신뢰성 높은 응답을 생성하는 기술

자율주행 로봇, ROS 시스템 로그 분석, 실시간 문제 해결 등에 RAG를 적극 활용 가능

동작방식
```
(1) 사용자의 질문 → (2) 관련 데이터 검색 → (3) 검색된 정보를 기반으로 LLM이 응답 생성
```
본 프로젝트에서의 활용 방향
``` 
(1) ROS 로그 및 센서 데이터 수집 → 데이터베이스에 저장
(2) LangChain의 벡터 데이터베이스 활용하여 검색
(3) 검색된 데이터를 LLM에게 제공 → 정확한 문제 해결 답변 생성
```
**어떤 상황(라이다, 주행, 맵 등), 어떤 테스크를 줬을 때 어떤 전략을 활용했는지 등을 llm이 검색을 통해 이해할 수 있게 규격화해서 저장하는 데이터베이스를 구축하여 활용하는 것이 좋을 듯**

+강화학습 등 머신러닝과 결합하여 그 데이터에 접근할 수 있도록 하는 것도 좋은 방향

### 2.1 SQL (Structured Query Language)
데이터베이스에서 데이터를 저장, 검색, 수정, 삭제하기 위해 사용되는 명령어이다.
주로 관계형 데이터베이스(RDBMS, Relational Database Management System) 에서 데이터를 관리하는 데 사용된다.

sql을 지원하는 대표적인 데이터베이스 시스템은 MySQL, PostgreSQL, SQLite, Microsoft SQL Server, Oracle Database등이 있다.

대표적인 명령어는 CRUD(Create, Read, Update, Delete) 연산이다.

---
**SQL 명령어 예시**

**(1) 데이터 검색**
```sql
SELECT name, age FROM users WHERE age > 20;
```
SELECT → 특정 컬럼 선택 (name, age)

FROM users → users 테이블에서 조회

WHERE age > 20 → 나이가 20보다 큰 데이터만 검색


**(2) 데이터 삭제**
```sql
DELETE FROM users WHERE name = 'Alice';
```
DELETE FROM users → users 테이블에서 데이터 삭제

WHERE name = 'Alice' → 'Alice' 데이터만 삭제



### 2.2 SQL을 활용한 LanChain + RAG활용 예시
```python
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.chains import SQLDatabaseChain

# 데이터베이스 연결
db = SQLDatabase.from_uri("sqlite:///users.db")

# LangChain LLM 설정
llm = ChatOpenAI(model_name="gpt-4", openai_api_key="YOUR_API_KEY")

# SQL 실행 체인 생성
db_chain = SQLDatabaseChain.from_llm(llm, db)

# SQL 쿼리 실행
query = "SELECT name FROM users WHERE age > 25;"
result = db_chain.run(query)

print("쿼리 결과:", result)
```



## 3. Agent 구축 공부


**작은 목표) 적절한 예시 찾은 후 공부하고 공유**

**나아간 목표) 예제를 본 프로젝트에 연관성 있게 조금 변형** 

**ex) 데이터베이스를 ros에서 주행기록이나 맵데이터로 구축하여 llm이 답변하도록 구현**

**-> 다음으로 어떤 기능 구현하면 좋을지 상담**

---
### 3.1 튜토리얼의 sql rag 활용 예시

본 프로젝트에서는 웹기반 rag를 쓸 필요성은 상대적으로 적다. 

이전에 받은 명령, 그 때의 맵 환경, 로봇의 위치, 경로 등의 데이터를 llm이 이해하기 쉬운 방식으로 데이터베이스화. 혹은 현재 수행중인 명령에 대한 로봇의 위치, 경로, 맵 데이터를 실시간으로 데이터베이스화하여,

rag를 통해 llm이 해당 데이터베이스를 검색하여 답변을 생성하도록 하는 방식이 보다 자율주행 에이전트 구축에 쓸모가 있을 것이라 판단하였다. 

따라서 sql 데이터베이스를 llm이 검색할 수 있도록 하는 에이전트를 예시로서 공부하겠다.

---
vs코드에 파이썬과 쥬피터 확장판 설치하여 활용

랭스미스 환경변수 설정 ~/.bashrc에 추가
```bash
export LANGSMITH_TRACING="true"

export LANGSMITH_API_KEY="..."
```
예시 에이전트 진행용 디렉토리 생성
```bash
mkdir sql_rag_agent
cd sql_rag_agent
```

해당 디렉토리에 파이썬 가상환경 생성 및 활성화, 비활성화
```bash
python3 -m venv venv

source venv/bin/activate

deactivate 
```
vs코드의 쥬피터 노트북 상에서 반드시 커널을 가상환경으로 맞추고 진행!

vs코드의 쥬피터 노트북 환경에서 사용할 환경변수 설정
```python
%%capture --no-stderr

%pip install --upgrade --quiet langchain-community langchainhub langgraph
```

```python
import os
import getpass

# LangSmith 사용을 원하지 않는 경우 아래 코드를 주석 처리하세요. 필수는 아닙니다.
if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
    os.environ["LANGSMITH_TRACING"] = "true
```
sqlite3 설치
```bash
sudo apt install sqlite3
```
sql_rag_agent 디렉토리에 chinook 데이터베이스 설치
```bash
curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db
```

데이터베이스와의 상호작용 (sql_rag_agent 디렉토리에서 실행)
```python
from langchain_community.utilities import SQLDatabase

# SQLite 데이터베이스 파일과 연결된 SQLDatabase 인스턴스 생성
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# 데이터베이스의 다이얼렉트 출력
print(db.dialect)

# 사용 가능한 테이블 이름 출력
print(db.get_usable_table_names())

# 'Artist' 테이블에서 상위 10개 행 선택
result = db.run("SELECT * FROM Artist LIMIT 10;")
#print(result) (원하면 #울 자우고 출력)
```

TypedDict 가져오고 입력, 새로 생성된 쿼리, 답변을 추적
```python
from typing_extensions import TypedDict


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
```
디렉토리의 가상환경 활성화 하고 채팅 모델 설치
```bash
pip install -qU "langchain[openai]"
```

오픈 ai 부르기
```python
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o", model_provider="openai")
```

프롬프트 허브에서 ㅍ롬프트 가져오기
```python
from langchain import hub

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

assert len(query_prompt_template.messages) == 1
query_prompt_template.messages[0].pretty_print()
```

모델에 프롬프트 제공
```python
from typing_extensions import Annotated


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}
```

#쿼리로 잘 변환해주는지 테스트 
```python
write_query({"question": "How many Employees are there?"})
```
쿼리 실행
```python
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool


def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}
```
#쿼리가 실행되는지 테스트
```python
execute_query({"query": "SELECT COUNT(EmployeeId) AS EmployeeCount FROM Employee;"})
```

베이터베이스의 정보 바탕으로 답변 생성
```python
def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}
```

랭그래프 활용

세 단계 (위에서 정의된 세 함수)를 단일 시퀀스로 만들고 컴파일
```python
from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()
```
제어 흐름 시각화
```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

애플리케이션 테스트
```python
for step in graph.stream(
    {"question": "How many employees are there?"}, stream_mode="updates"
):
    print(step)
```
**간단한 sql rag 체인 구현완료!**

---
### 3.2 보다 발전된 에이전트
데이터베이스에 접근해서 새로운 테이블 읽고 필요한 정보만 빼서 새로 테이블 만들어정리 ->

답변 생성?

기본키 - 테이블에서 각 행(레코드)을 고유하게 식별하는 컬럼(또는 컬럼 조합)

왜래키 - 다른 테이블의 기본키(PK)를 참조하여 테이블 간 관계를 맺는 키

#### 3.2.1 create_react_agent 
- llm을 거쳐 툴과 유저가 상호작용할 수 있도록 사전에 만들어진 에이전트

활용 코드 예시
```python
>>> from datetime import datetime
>>> from langchain_openai import ChatOpenAI
>>> from langgraph.prebuilt import create_react_agent


... def check_weather(location: str, at_time: datetime | None = None) -> str:
...     '''Return the weather forecast for the specified location.'''
...     return f"It's always sunny in {location}"
>>>
>>> tools = [check_weather]
>>> model = ChatOpenAI(model="gpt-4o")
>>> graph = create_react_agent(model, tools=tools)
>>> inputs = {"messages": [("user", "what is the weather in sf")]}
>>> for s in graph.stream(inputs, stream_mode="values"):
...     message = s["messages"][-1]
...     if isinstance(message, tuple):
...         print(message)
...     else:
...         message.pretty_print()
```

check_wether라는 정의된 함수와 상호작용하는 모습을 확인할 수 있다.
```
('user', 'what is the weather in sf')
================================== Ai Message ==================================
Tool Calls:
check_weather (call_LUzFvKJRuaWQPeXvBOzwhQOu)
Call ID: call_LUzFvKJRuaWQPeXvBOzwhQOu
Args:
    location: San Francisco
================================= Tool Message =================================
Name: check_weather
It's always sunny in San Francisco
================================== Ai Message ==================================
The weather in San Francisco is sunny.
```

