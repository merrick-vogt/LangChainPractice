import os
import openai
import sys
from langchain_community.document_loaders.text import TextLoader

from langchain.indexes import VectorstoreIndexCreator
from langchain_community.llms.openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import constants

os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

query = sys.argv[1]
loader = TextLoader('data.txt')
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query))

