# Legal Document Analysis using Pinecone, OpenAI, and LangChain

This repo contains a program for analyzing the actual depositions to identify instances of admissions and contradictions by the witness. It uses **Pinecone**, **OpenAI**, and **LangChain**.

## What is a Deposition?

A deposition is a legal terminology. In simple words, Deposition is a written statements of witnesses taken before a magistrate or other judicial authority. To study more about the depositions, please see the detailed [page](https://en.wikipedia.org/wiki/Deposition_(law)).

## What is an Admission

An admission in a deposition is when someone, during a legal deposition, confirms or agrees with a fact or a statement that is presented. This 'admission' is crucial because, in a legal context, it can be used as evidence in court. The person cannot later deny that fact or statement, as it is now on the record. So, in simpler terms, it's like saying 'yes, that's true' during a serious question-and-answer session that can be used in court.

## What is a Contradiction
    
A contradiction in a deposition refers to when a person states something that opposes or is inconsistent with something they've previously said during the same deposition or in previous depositions or statements. This could be like saying it's raining outside, and later saying it's sunny, without a reasonable explanation for the discrepancy. In a legal context, contradictions can weaken a person's credibility and affect the overall outcome of the case as they suggest that the person might not be telling the truth or their memory could be unreliable.

## Overview

This section provides a brief overview of the whole process which involves processing a PDF document containing excerpts from a legal deposition, identification of relevant sections, and saving the analysis results in CSV files. The main steps are as follows:

1. **Load PDF Document**: The PDF document is placed in the `Documents` directory and read using the `PyPDFLoader` API from LangChain.
2. **Split Document**: The large document is split into smaller chunks for easier processing.
3. **Create Embeddings**: Embedding vectors are created using OpenAI's `text-embedding-ada-002` model through the `OpenAIEmbeddings` high-level API.
4. **Store Embeddings**: These embeddings are upserted (saved) into the **Pinecone** vector database.
5. **Identify Instances**: 
    - The relevant embeddings for admissions and contradictions are retrieved from the **Pinecone** database based on custom prompts.
    - A **Retrieval Chain** is used to find exact instances using ChatGPT.
6. **Save Results**: The final results of all the occurrences of admissions and contradictions are stored as CSV files in the `Admissions` and `Contradictions` directories, respectively.

## Directory Structure

- `Documents/`: Contains the PDF document with the deposition excerpts.
- `Admissions/`: Directory where the CSV files with instances of admissions are stored.
- `Contradictions/`: Directory where the CSV files with instances of contradictions are stored.
- `Modal/`: Directory which contains the Modal App script to run the same code in the `Modal` cloud.

## Dependencies

The Python scripts in this repo require Python **3.9**. Additionally, the list of required libraries along with their versions is present in the `requirements.txt` file.

## Usage

### Simple Python script running locally

- Place the PDF document in the `Documents` directory.
- Place `OPENAI_API_KEY` and `PINECONE_API_KEY` in the `.env` file.
- Run the `Python_Script.ipynb` notebook using the command:

```
cd <path_where_you_clones_this_repo>
pip install -r requirements.txt
jupyter notebook Python_Script.ipynb
```

- This will open the notebook in your browser. You can run it and see the real-time execution.

### Modal App

To run the same code in the cloud, a Modal App has also been created. In order to use this app, please store the:

* **PINECONE_API_KEY** as `my-pinecone-secret` in Modal account
* **OPENAI_API_KEY** as `my-openai-secret` in Modal account
    
Then, run the following in the command line:

```
cd <path_where_you_clones_this_repo>
cd Modal
modal run Modal_App.py
```

## Example
An example of the output CSV files:

### Admissions

| Topic                                | Content                                                                                                                                       | Reference          | Reason                                                                                                                      |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|--------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Recollection of seeing the cutout     | I saw it go out.                                                                                                                              | line 24, page 192  | The witness initially mentioned it was an assumption, but later admitted to seeing the cutout.                              |
| Observation of water changing color   | Yes.                                                                                                                                          | line 15, page 170  | The witness confirmed observing the water change color while working at Metro-Atlantic.                                     |

#### The screenshot of the CSV file:

<img src="https://github.com/MUmairAB/Analyzing-Legal-Documents-using-ChatGPT/blob/main/Images/Admissions%20CSV%20file.png?raw=true"/>

## Contradictions

| Topic                   | Assertion Content                                      | Assertion Reference | Contradiction Content                                                      | Contradiction Reference | Reason                                                                                                                                        |
|-------------------------|--------------------------------------------------------|---------------------|---------------------------------------------------------------------------|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Knowledge of Ownership  | Did not know who owned what at that time.              | line 17, page 150   | Did not know where any boundaries of the land were as between ownership of one person or another. | line 20, page 150         | The witness initially stated he did not know who owned what, but later contradicted this by saying he did not know where the boundaries of the land were between different owners. |
| Orientation to Map      | Recognized features east and west of the facilities.   | line 19, page 218   | Mentioned never looking at the features.                                   | line 23, page 218         | The witness first acknowledged recognizing features east and west of the facilities on the map, but later contradicted this by stating that he never looked at those features.     |

#### The screenshot of the CSV file:

<img src="https://github.com/MUmairAB/Analyzing-Legal-Documents-using-ChatGPT/blob/main/Images/Contradictions%20CSV%20file.png?raw=true" />


# Contributing

For any contributions to improve this project, please create a [pull request](https://github.com/MUmairAB/Analyzing-Legal-Documents-using-ChatGPT/tree/main).
