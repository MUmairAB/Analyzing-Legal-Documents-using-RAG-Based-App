"""
This script creates a Modal App for the same task. This means that this script also creates a RAG application but the only difference is that this app runs in the Modal cloud rather than on a local machine.

Overview:
The script analyzes legal documents using Pinecone, OpenAI, and LangChain. It processes a PDF document containing excerpts of a deposition placed in the Documents directory. The workflow is as follows:

Document Reading:

The script uses the PyPDFLoader API from LangChain to read the document locally in the main() function. Note: Since Modal requires local data as function inputs, refer to the Modal documentation "https://modal.com/docs/guide/local-data" for more details.

Document Splitting:
The script splits the large document into smaller chunks using the split_the_local_document() method. This method runs in the Modal cloud and returns LangChain Document objects.

Embedding Creation and Storage:
The subsequest step involves using the upsert_vectors_to_pinecone_from_document_object() method to create embeddings of these smaller chunks and store them in the Pinecone database.

This step is also executed in the Modal cloud.

Finding Admissions:
The find_admissions() method is executed in the Modal cloud. It uses another method called QnAchain() which retrieves the relevant data from the Pinecone and feeds it to the ChatGPT to get all instances of admissions. It returns the output as a dictionary.

The script then runs the create_json() method locally to store the results in the local Admissions directory.

Finding Contradictions:
To find instances of contradictions, the find_contradictions() method is run in the Modal cloud. It also uses the same method called QnAchain() which retrieves the relevant data from the Pinecone and feeds it to the ChatGPT to get all instances of ontradictions(). It returns the output as a dictionary.

The script subsequently runs the create_json() method locally to store the results in the local Contradictions directory.

note: To use this script, store the:
    - PINECONE_API_KEY as "my-pinecone-secret" in Modal account
    - OPENAI_API_KEY as "my-openai-secret" in Modal account
"""
#Import necessary libraries
import os
import sys
import json
import glob
import time
######    LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import SentenceTransformerEmbeddings

######    Pinecone
from langchain_pinecone import PineconeVectorStore #High-level API from langchain
from pinecone import Pinecone, ServerlessSpec
######    OpenAI
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
######    Modal
import modal


#Create the custom image for our required packages
my_image = modal.Image.debian_slim(python_version="3.9").pip_install(
    "pinecone-client==3.2.2",
    "openai==1.30.1",
    "langchain==0.2.0",
    "tiktoken==0.7.0",
    "langchain-community==0.2.0",
    "langchain-text-splitters==0.2.0",
    "langchain-openai==0.1.7",
    "langchain-pinecone==0.1.1",
    "python-dotenv==0.21.0",
    "langchain-text-splitters==0.2.0",
    "pypdf==4.2.0",
    "pypdf2==3.0.1"
)

#Instantiate the modal App with custom image and secrets
app = modal.App(name="legal-docs-analyzer",
                image=my_image,
                secrets=[modal.Secret.from_name("my-openai-secret"), #Pass the OpenAI API key as a Modal secret to this function
                       modal.Secret.from_name("my-pinecone-secret") #Pass the Pinecone API key as a Modal secret to this function
                ]
                )

@app.function()
def split_the_local_document(local_doc) -> list:
    """
        This method accepts a PyPDFLoader object which contains the
        contents of the local PDF file. It simply splits the document into
        smaller chunks, and returns these chunks after adding the page number.

        Inputs: 
            - local_doc
                A PyPDFLoader object which contains the contents of the local PDF file
        Outputs:
            - doc_chunks : list[langchain_core.documents.base.Document]
                A list of LangChain Document objets created from the main PDF file.
    """
        
    """
    Since our document is quite large, we need to split it into 
    smaller chunks.
    """
    #Instantiate the RecursiveCharacterTextSplitter API
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap  = 200, #Set the overlap of chunks for better results
    )
    #Split the read document (langchain PyPDFLoader object)
    doc_chunks = text_splitter.split_documents(local_doc)
    """
    Since we also need to specify the page number in our answer, 
    we'll have to append the page number at the end of each chunk.
    """
    #Add the page number
    for i in range(len(doc_chunks)):
        doc_chunks[i].page_content = doc_chunks[i].page_content + f"\n\npage: " + str(doc_chunks[i].metadata["page"] + 146)
    
    return doc_chunks


def create_json(data_str: str, directory_name: str) -> dict:
    """
        This function accepts a Python dictionary as a string,
        converts it into a normal dictionary, and saves it in the local
        directory specified in directory_name. Finally it returns the dictionary.

        Inputs:
            - data_str : str
                A Python dictionary in the form of a string
            - directory_name : str
                A string specifying the name of the parent directory where the file will be stored.
        Outputs:
            - results_dict : dict
                A dictionary created from the input sting.
    """
    #Convert the resultant string object to Python dictionary
    try:
        results_dict = json.loads(data_str)
        #Save the resultant dictionary as a JSON file
        with open(os.path.join(directory_name,"result.json")) as f:
            json.dump(results_dict[directory_name.lower()], f)
        return results_dict

    except json.JSONDecodeError as e:
        print(e)
        return {"error":e} # Return an empty dictionay


@app.function() 
def upsert_vectors_to_pinecone_from_document_object(doc_chunks: list) -> bool:
    """
        This method accepts a LangChain Document object created
        by splitting up the large document in the form of a list.
        Then, it creates the embedding vectors using "text-embedding-ada-002"
        model from OpenAI. Finally, it upserts / saves these embeddings
        in the Pincecone index in the {NAMESPACE} namespace.

        Inputs: 
            - doc_chunks : list[langchain_core.documents.base.Document]
                The list of Document objects created by splitting the large documents into smaller chunks using PyPDFLoader.
        Outputs: 
            - bool
                True if upsert operation in Pinecone index is a success, otherwise, False.
    """
    index_name = "modal-legal-index" #A variable to store the index name for the Pinecone
    NAMESPACE = "doc1" #A variable to store the namespace to store our embeddings

    #Configure the client connection to Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    """
    We'll check if an index already exists with the
    name "modal-legal-index". If it does, we'll delete it
    and create a new one. Because we'are only allowed to 
    create 5 indexes in Starter plan.
    """        
    
    #If this index already exists, we'll delete it
    if index_name in pc.list_indexes().names():  
        pc.delete_index(index_name)  
    #Create a new index  
    pc.create_index(
        index_name,
        dimension= 1536, # dimensionality of text-embedding-ada-002
        metric='dotproduct',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )  
    #Wait for index to be initialized  
    while not pc.describe_index(index_name).status['ready']:  
        time.sleep(1)
    
    #Instantiate the OpenAI Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    #Create the embeddings for each chunk of the document and store it in the Pinecone index in the namespace
    try:
        vectorstore_from_docs = PineconeVectorStore.from_documents(
            doc_chunks,
            index_name=index_name,
            embedding=embeddings,
            namespace=NAMESPACE
        )
        return True
    except Exception as e:
        print(f"OpenAI API returned an API Error: {e}")
        return False

def QnAchain(query: str, top_k: int) -> dict:
    """
        This method receieves a query, and then retrieves the relevant
        data from the Pinecone through a Retrieval. It then, feeds this
        data to ChatGPT through the Retrieval Chain from LangChain to
        get the relevant answer for the input query. Finally. it
        returns the ChatGPT response.

        Inputs:
            - query : str
                The custom prompt to detect either the instances of admissions or contradictions.
            - top_k : int
                The number of maximum relevant documents to be retrieved from the Pinecone
        
        Outputs:
            - dict()
                A dictionary containing the response from ChatGPT
    """

    index_name = "modal-legal-index" #A variable to store the index name for the Pinecone
    NAMESPACE = "doc1" #A variable to store the namespace where our embeddings are stored
    
    #Instantiate the OpenAI Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key= os.environ["OPENAI_API_KEY"])

    #Instantiate the LLM
    llm = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

    #Initialize the PineconeVectorStore object to retrieve the data for our query
    vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding= embeddings,
            namespace=NAMESPACE
        )
    
    #Instantiate the retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
                                        search_type="mmr", #Maximal Marginal Relevance
                                        #Let's return maximum 8 relevant documents
                                        #lambda_mult sets the diversity of results returned by MMR
                                        search_kwargs={'k': top_k, 'lambda_mult': 0.25}
                                        )
                                )
    
    #Let's get the instances of admissions
    try:
        GPT_response = qa_chain.invoke(query)
        return GPT_response
    except Exception as e:
        print(e)
        return {"error":e}

@app.function()
def find_admissions() -> dict:
    """
        This method declares a well-written custom prompt, and then
        uses the QnAchain() method to find all the instances of admissions,
        and simply returns the result.

        Inputs:
            - None
        Output:
            - dict()
                A dictionary containing the response from ChatGPT
    """
    admissions_prompt = """
        You are a legal expert analyzing a witness deposition. Your task is to identify all instances of "admissions" in the text. For each instance, provide the line number(s) and explain why it is considered an "admission".

        Definitions:
        "Admissions": Statements where the witness acknowledges or agrees to a fact, confirms involvement, or accepts responsibility. Examples include phrases like "I admit", "I agree", "Yes, that's true", "I acknowledge that", "I suppose so.", "I guess that's right.", "You could say that.", "I think so.", "To the best of my knowledge...", "I recognize this document.", "I received the email.", "I attended the meeting." etc.

        Please analyze the following deposition text and identify any instances of admissions. At the start of each line, the line number is given, and at the end of the page, a page number is also given as "page: ". 

        Provide the output in the following Python dictionary format. Please make sure that Python dictionary is well structured.:

        {"admissions": [ {"topic": "Brief title of admission","content": "Content of the admission","reference": "line x, page y","reason": "Explanation of why this instance is considered an admission"},...]}
    """

    #return admissions results
    return QnAchain(query=admissions_prompt, top_k=8) #For admissiosn, we have set the retrieval to terieve maximum 8 relevant documents

@app.function()
def find_contradictions() -> dict:
    """
        This method declares a well-written custom prompt, and then
        uses the QnAchain() method to find all the instances of contradictions,
        and simply returns the result.

        Inputs:
            - None
        Output:
            - dict()
                A dictionary containing the response from ChatGPT
    """
    
    contradictions_prompt = """
    You are a legal expert analyzing a witness deposition. Your task is to identify instances of "contradictions" in the text. For each instance, provide the line number(s) and explain why it is considered a "contradiction".

    Definitions:
    "Contradictions": Statements where the witness's current statement conflicts with a previous statement or established fact. Examples include changing answers, denying previous acknowledgments, or providing conflicting information.

    Please analyze the following deposition text and identify any instances of contradictions. At the start of each line, the line number is given, and at the end of the page, a page number is also given as "page: ". 

    Provide the output in the following dictionary format. Please make sure that the dictionary is well structured.:

    {"contradictions": [ {"topic": "Brief title of contradiction","assertion_content": "Content of the initial assertion","assertion_reference": "line x, page y","contradiction_content": "Content of the contradictory statement","contradiction_reference": "line x, page y","reason": "Explanation of why this instance is considered a contradiction"},...]}
    """

    #return contradictions results
    return QnAchain(query=contradictions_prompt, top_k=5) #For admissiosn, we have set the retrieval to terieve maximum 5 relevant documents

@app.local_entrypoint()
def main():
    
    #Get the list of PDF files present in the local "Documents" directory
    path_list = glob.glob(os.path.join("Documents","*.PDF"))
    #Instantiate the PyPDFLoader
    loader = PyPDFLoader(path_list[0])
    #Load the file contents
    doc = loader.load()
    #Split the locally read document into smaller chunks in the Modal cloud
    doc_chunks = split_the_local_document.remote(doc)

    #Create the embeddings of the document chuks and save them to Pinecone
    if not upsert_vectors_to_pinecone_from_document_object.remote(doc_chunks):
        sys.exit("There was error in \"upsert_vectors_to_pinecone_from_document_object\" function!")
    else:
        print("The embeddings of the document have successfully been saved in Pinecone")

    #Let's introduce some delay as a precaution, to make sure we
    # don't breach the OpenAI API limit.
    time.sleep(60)

    #Find the instances of admissions
    admissions_results = find_admissions.remote()

    #Check if there was any error in the find_admissions() call
    if "error" in admissions_results.keys():
        sys.exit("There was error in \"find_admissions\" function:",admissions_results["error"])

    #Save the admissions result as JSON file locally
    else:
        admissions_results_dict = create_json(data_str=admissions_results["result"], directory_name="Admissions")
        if not admissions_results_dict:
            print("No instances of Admission found!")
        elif "error" in admissions_results_dict.keys():
            print("There was an error while save the admissions results as a JSON file:",admissions_results_dict["error"])
        else:
            print("All the instances of Admissions have been saved in the \"Admissions\" directory!")

    #Let's introduce some delay as a precaution, to make sure we
    # don't breach the OpenAI API limit.
    time.sleep(60)

    #Find the instances of contradictions
    contradictions_results = find_contradictions()

    #Check if there was any error in the find_admissions() call
    if "error" in contradictions_results.keys():
        sys.exit("There was error in \"find_contradictions\" function:",contradictions_results["error"])

    #Save the contradictions result as JSON file locally
    else:
        contradictions_results_dict = create_json(data_str=contradictions_results["result"], directory_name="Contradictions")
        if not contradictions_results_dict:
            print("No instances of Contradictions found!")
        elif "error" in contradictions_results_dict.keys():
            print("There was an error while save the contradictions results as a JSON file:",contradictions_results_dict["error"])
        else:
            print("All the instances of Contradictions have been saved in the \"Contradictions\" directory!")
