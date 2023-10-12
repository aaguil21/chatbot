import os
from apikey import apikey

os.environ['HUGGINGFACEHUB_API_TOKEN'] = apikey

import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

#App framework
st.title('Youtube GPT Creator')
prompt = st.text_input('Plug in your prompt here')

title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Write me a youtube video title about the topic:{topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template = 'Write a 3 paragraph youtube video script on this title: {title} while leveraging this wikipedia research: {wikipedia_research}'
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key ='chat_history')
script_memory= ConversationBufferMemory(input_key='title', memory_key ='chat_history')

# Llms 

llm = HuggingFaceHub(repo_id = 'google/flan-t5-xxl', model_kwargs={'temperature':0.2, 'max_new_tokens':200})
title_chain = LLMChain(llm = llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm = llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

#Show stuff to the screen if there's a prompt
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)
