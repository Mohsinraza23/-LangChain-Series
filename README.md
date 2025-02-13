# -LangChain-Series

# LangChain Series

Welcome to the **LangChain Series**! This series is designed to guide you through using **LangChain**, a powerful library for building applications with large language models (LLMs) like OpenAI, Gemini, and more. Whether you are a beginner or an experienced developer, you'll find valuable insights here to enhance your projects with LangChain.

## Overview
LangChain is an open-source framework that simplifies the integration of large language models into various applications. With LangChain, you can create robust AI-driven solutions with ease, including chatbots, document analysis tools, and more.

## What You'll Learn
This series will cover:
- Setting up LangChain and LLMs
- Integrating with popular models like OpenAI GPT and Gemini
- Using advanced LangChain features like chains, agents, and tools
- Building end-to-end AI-powered applications
- Best practices for working with LangChain

## Key Concepts
- **Chains**: Chains allow you to sequence multiple prompts or actions together. You can build complex workflows by chaining various LLMs or tools.
- **Agents**: Agents use an LLM to choose the best course of action, such as calling APIs or accessing external resources.
- **Tools**: Tools in LangChain provide the ability to interact with external systems, APIs, or databases.
- **Memory**: LangChain supports memory features, which help you create persistent stateful models that remember past interactions.

## Example
Here’s a simple example of using LangChain with OpenAI:

```python
from langchain.llms import OpenAI

# Initialize the OpenAI model
llm = OpenAI(model="text-davinci-003")

# Generate text
response = llm("What is LangChain?")
print(response)
```

## Resources
- [LangChain Documentation](https://python.langchain.com)
- [LangChain GitHub](https://github.com/hwchase17/langchain)
- [OpenAI API Documentation](https://beta.openai.com/docs/)
- [LangChain Community GitHub](https://github.com/langchain-ai/langchain)

## Contributions
Contributions are always welcome! If you have suggestions, improvements, or additional examples, feel free to open an issue or submit a pull request.

---

# LangChain Series - Class 01

## Introduction
Welcome to the first class of the **LangChain Series**! In this session, we set up a Python-based project using LangChain and Streamlit, leveraging the OpenAI API to build a simple demo application.

## Requirements
- Anaconda installed on your system.
- Python 3.13.1 or compatible.
- OpenAI API Key.

## Setup Instructions
### Step 1: Create and Activate Environment
Open Command Prompt (CMD) and run the following commands:

```sh
conda create -p myenv python=3.13.1 -y
conda activate myenv
```

### Step 2: Prepare `requirement.txt`
Create a file named `requirement.txt` with the following content:

```
openai
langchain
langchain_community
streamlit
```

Install the required packages:

```sh
pip install -r requirement.txt
```

### Step 3: Add Constants
Create a file named `constants.py` with the following content:

```python
openai_key = 'Your API Key'
```

Replace `'Your API Key'` with your actual OpenAI API key.

### Step 4: Create Main Application
Create a file named `main.py` with the following code:

```python
import os
from constants import openai_key
from langchain.llms import OpenAI
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework
st.title('Langchain Demo With OPENAI API')
input_text = st.text_input("Search the topic you want")

# OpenAI LLMs
llm = OpenAI(temperature=0.8)

if input_text:
    st.write(llm(input_text))
```

### Step 5: Run the Application
Run the following command in your terminal:

```sh
streamlit run main.py
```

Open the URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.

## Features
✅ **Interactive Input**: Users can type a query into the input box.
✅ **LLM Response**: The application uses LangChain and OpenAI's API to generate a response based on the input.

## Notes
- Ensure your OpenAI API key has sufficient credits to test the application.
- Enjoy building intelligent applications with LangChain! 😊



LangChain Series - Class 02
Introduction
Welcome to the second class of the LangChain Series! In this session, we build upon the foundations established in Class 1. We'll focus on creating more advanced LangChain-based applications by integrating multiple prompt templates, sequential chains, and memory buffers. The goal is to develop a celebrity search application that showcases LangChain's capabilities in dynamic prompt management and interactive conversation.

Step 1: Creating practice.py
Start by creating a new Python file called practice.py. This will be where you experiment with LangChain concepts.

# practice.py
import os
from constants import openai_key
from langchain.llms import OpenAI
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework
st.title('Langchain Demo With OPENAI API')
input_text = st.text_input("Enter Anything You Want To Ask?")


# OpenAI LLMs
llm = OpenAI(temperature=0.8)

if input_text:
    st.write(chain.run(input_text))
Step 2: Building the Celebrity Search Application
Now we’ll create a prompt template for searching celebrity details.

# practice.py (Step 2)
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework
st.title('Langchain Demo With OPENAI API')
input_text = st.text_input("Enter Anything You Want To Ask?")

# Prompt Template for celebrity search
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

# OpenAI LLMs
llm = OpenAI(temperature=0.8)
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True)

if input_text:
    st.write(chain.run(input_text))
Output Section
Asking -
STEP-TWO-1
Output -
STEP-TWO-2
Step 3: Combining Multiple Prompt Templates
In this step, we’ll combine two different prompt templates to handle multiple queries.

# practice.py (Step 3)
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework
st.title('Langchain Demo With OPENAI API')
input_text = st.text_input("Enter Anything You Want To Ask?")

# Prompt Template for celebrity search
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

# Prompt Template for birth date
second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born?"
)

# OpenAI LLMs
llm = OpenAI(temperature=0.8)
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person')
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob')

if input_text:
    st.write(chain.run(input_text))
    st.write(chain2.run(chain.run(input_text)))
Output Section
Asking -
STEP-THREE-1
Output -
STEP-THREE-2
Step 4: Switching to SequentialChain
In this step, we use a SequentialChain to process multiple prompts in sequence.

# practice.py (Step 4)
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework
st.title('Langchain Demo With OPENAI API')
input_text = st.text_input("Enter Anything You Want To Ask?")

# Prompt Template for celebrity search
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

# Prompt Template for birth date
second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born?"
)

# OpenAI LLMs
llm = OpenAI(temperature=0.8)
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person')
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob')

parent_chain = SequentialChain(
    chains=[chain, chain2],
    input_variables=['name'],
    output_variables=['person', 'dob'],
    verbose=True
)

if input_text:
    st.write(parent_chain({'name': input_text}))
Output Section
Output -
STEP-FOUR-1
Step 5: Adding a Third Prompt Template
We add a third template to include even more detailed information.

# practice.py (Step 5)
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework
st.title('Langchain Demo With OPENAI API')
input_text = st.text_input("Enter Anything You Want To Ask?")

# Prompt Template for celebrity search
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

# Prompt Template for birth date
second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born?"
)

# Prompt Template for major events
third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events that happened around {dob} in the world."
)

# OpenAI LLMs
llm = OpenAI(temperature=0.8)
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person')
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob')
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='description')

parent_chain = SequentialChain(
    chains=[chain, chain2, chain3],
    input_variables=['name'],
    output_variables=['person', 'dob', 'description'],
    verbose=True
)

if input_text:
    st.write(parent_chain({'name': input_text}))
Output Section
Asking -
STEP-FIVE-1
Output -
STEP-FIVE-2
Step 6: Adding Memory Buffers
Now we’ll add memory buffers to retain the conversation context between each step.

# practice.py (Step 6)
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework
st.title('Celebrity Search Application')
input_text = st.text_input("Enter Anything You Want To Ask?")

# Prompt Template for celebrity search
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

# Prompt Template for birth date
second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born?"
)

# Prompt Template for major events
third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events that happened around {dob} in the world."
)

# Memory Buffers
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='dob_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# OpenAI LLMs
llm = OpenAI(temperature=0.8)
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person', memory=person_memory)
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob', memory=dob_memory)
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='description', memory=descr_memory)

# Sequential chain
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3],
    input_variables=['name'],
    output_variables=['person', 'dob', 'description'],
    verbose=True
)

if input_text:
    st.write(parent_chain({'name': input_text}))

    # Expander for memory history
    with st.expander('Person History'):
        st.info(person_memory.buffer)

    with st.expander('Major Events History'):
        st.info(descr_memory.buffer)
Output Section
Output -
STEP-SIX-1
Features
Prompt Templates: Create dynamic and customized prompts for different tasks.
Sequential Chains: Process multiple prompts in sequence for complex workflows.
Memory Buffers: Retain context to enhance the user experience in interactive applications.
Notes
Remember to replace openai_key in constants.py with your OpenAI API key.
You can adjust the prompt templates based on your needs.
The memory buffers allow you to track and utilize prior interactions for more meaningful results.
Enjoy building intelligent applications with LangChain! 😊
