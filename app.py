import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader

##ssstreamlit APP
st.set_page_config(page_title="Langchain:Summarize Text From YT or website",page_icon="emoji")
st.title("Langchain:Summarize Text YT or website")
st.subheader("Summarize URL")


## get the Groq API key and url(YT or website)to be summarized
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")


#Gemma Model  Using Groq Api
llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

prompt_template="""
Provide a summary of the follwing content in 300 words
Content:{text}
"""

prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("summarize the content from Youtube or website"):
    ##validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get the started")
    elif not  validators.url(generic_url):
        st.error("Please enter a valid url.It can may a YT video url or website url")
    
    else:
        try:
            with st.spinner("waiting...."):
                #loading the website or YT video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) Applewebkit/537.36 (KHTML,like Gecko) chrome/116.0.0.0 Safari/537.36"})
                    
                docs=loader.load()
                ##chain for summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)


                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")




