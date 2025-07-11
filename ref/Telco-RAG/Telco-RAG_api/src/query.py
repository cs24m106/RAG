import os
import json
import numpy as np
import torch
import traceback
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset
from src.retrieval import find_nearest_neighbors_faiss
from src.index import get_faiss_batch_index
from src.online_retrieval.pdf_reader import fetch_snippets_and_search
from src.embeddings import get_embeddings
from src.get_definitions import define_TA_question
from src.input import get_documents
from src.chunking import chunk_doc
from src.LLMs.LLM import submit_prompt_flex, a_submit_prompt_flex, embedding
from src.validator import validator_RAG
from src.NNRouter import NNRouter
from api.LLM import a_submit_prompt_flex_UI, submit_prompt_flex_UI
import logging

resources_path = os.path.join(os.path.dirname(__file__), 'resources')
logger = logging.getLogger(__name__) # Setup logging

class Query:
    def __init__(self, query, context):
        self.question = query
        self.query = query 
        self.enhanced_query = query
        self.context = [context] if isinstance(context, str) else context
        self.context_source = []
        self.wg = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NNRouter()
        self.model.load_state_dict(torch.load(os.path.join(resources_path, 'router_new.pth'), map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()
        self.original_labels_mapping = np.arange(21, 39)

    def def_TA_question(self):
        self.query = define_TA_question(self.query)
        self.enhanced_query = self.query

    def candidate_answers(self, model_name='gpt-4o-mini', UI_flag=True):
        row_context = f"""
Provide all the possible answers to the following question considering your knowledge and the text provided.
Question: {self.query}
Considering the following context:
{self.context}
Provide all the possible answers to the following question considering your knowledge and the text provided.
Question: {self.question}
Ensure none of the answers provided contradicts your knowledge and each answer has at most 100 characters.
        """
        logger.info("generating candidate answers based on retrieved content. prompt-format: <--ins-->[query] <--use-->[content] <--gen-->[question]")
        try:            
            generated_output_str = submit_prompt_flex_UI(row_context, model=model_name) if UI_flag else submit_prompt_flex(row_context, model_name=model_name)
            if generated_output_str != "NO": 
                self.context = generated_output_str 
                logger.info(f"generated context: {repr(self.context)}")
                self.enhanced_query = self.query + '\n' + self.context
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    @staticmethod
    def get_embeddings_list(text_list):
        embeddings = embedding(text_list)
        logger.info(f"text_list size:({len(text_list)}) :: embedding dims: {embeddings.shape}")
        return dict(zip(text_list, embeddings))

    @staticmethod
    def inner_product(a, b):
        return sum(x * y for x, y in zip(a, b))
    
    @staticmethod
    def get_col2(embeddings_list, reset=False):
        file_path = os.path.join(resources_path, 'series_description.json')
        if reset and os.path.isfile(file_path):
            os.remove(file_path)
            
        if os.path.isfile(file_path):
            logger.info("loading series description from existing file")
            with open(file_path, 'r') as file:
                series_dict = json.load(file)
        else:
            logger.info("creating new series description from pre-existing knowledge base")
            topics_with_series = [
                ("Requirements (21 series): Focuses on the overarching requirements necessary for UMTS (Universal Mobile Telecommunications System) and later cellular standards, including GSM enhancements, security standards, and the general evolution of 3GPP systems.", "21 series"),
                ("Service aspects ('stage 1') (22 series): This series details the initial specifications for services provided by the network, outlining the service requirements before the technical realization is detailed.", "22 series"),
                ("Technical realization ('stage 2') (23 series): Focuses on the architectural and functional framework necessary to implement the services described in stage 1, providing a bridge to the detailed protocols and interfaces defined in stage 3.", "23 series"),
                ("Signalling protocols ('stage 3') - user equipment to network (24 series): Details the protocols and signaling procedures for communication between user equipment and the network, ensuring interoperability and successful service delivery.", "24 series"),
                ("Radio aspects (25 series): Covers the specifications related to radio transmission technologies, including frequency bands, modulation schemes, and antenna specifications, critical for ensuring efficient and effective wireless communication.", "25 series"),
                ("CODECs (26 series): Contains specifications for voice, audio, and video codecs used in the network, defining how data is compressed and decompressed to enable efficient transmission over bandwidth-limited wireless networks.", "26 series"),
                ("Data (27 series): This series focuses on the data services and capabilities of the network, including specifications for data transmission rates, data service features, and support for various data applications.", "27 series"),
                ("Signalling protocols ('stage 3') - (RSS-CN) and OAM&P and Charging (overflow from 32.- range) (28 series): Addresses additional signaling protocols related to operation, administration, maintenance, provisioning, and charging, complementing the core signaling protocols outlined in the 24 series.", "28 series"),
                ("Signalling protocols ('stage 3') - intra-fixed-network (29 series): Specifies signaling protocols used within the fixed parts of the network, ensuring that various network elements can communicate effectively to provide seamless service to users.", "29 series"),
                ("Programme management (30 series): Relates to the management and coordination of 3GPP projects and work items, including documentation and specification management procedures.", "30 series"),
                ("Subscriber Identity Module (SIM / USIM), IC Cards. Test specs. (31 series): Covers specifications for SIM and USIM cards, including physical characteristics, security features, and interaction with mobile devices, as well as testing specifications for these components.", "31 series"),
                ("OAM&P and Charging (32 series): Focuses on operation, administration, maintenance, and provisioning aspects of the network, as well as the charging principles and mechanisms for billing and accounting of network services.", "32 series"),
                ("Security aspects (33 series): Details the security mechanisms and protocols necessary to protect network operations, user data, and communication privacy, including authentication, encryption, and integrity protection measures.", "33 series"),
                ("UE and (U)SIM test specifications (34 series): Contains test specifications for User Equipment (UE) and (U)SIM cards, ensuring that devices and SIM cards meet 3GPP standards and perform correctly in the network.", "34 series"),
                ("Security algorithms (35 series): Specifies the cryptographic algorithms used in the network for securing user data and signaling information, including encryption algorithms and key management procedures.", "35 series"),
                ("LTE (Evolved UTRA), LTE-Advanced, LTE-Advanced Pro radio technology (36 series): Details the technical specifications for LTE, LTE-Advanced, and LTE-Advanced Pro technologies, including radio access network (RAN) protocols, modulation schemes, and network architecture.", "36 series"),
                ("Multiple radio access technology aspects (37 series): Addresses the integration and interoperability of multiple radio access technologies within the network, enabling seamless service across different types of network infrastructure.", "37 series"),
                ("Radio technology beyond LTE (38 series): Focuses on the development and specification of radio technologies that extend beyond the capabilities of LTE, aiming to improve speed, efficiency, and functionality for future cellular networks.", "38 series")
            ]
            series_dict = {index: {"description": desc, "embeddings": Query.get_embeddings(desc)} for desc, index in topics_with_series}
            with open(file_path, 'w') as file:
                json.dump(series_dict, file, indent=4)
        
        logger.debug(f"no.of series_dict entries: {len(series_dict)} | indices: {list(series_dict.keys())} | embedding dims: {[len(series_dict[idx]['embeddings']) for idx in series_dict]}")
        similarity_column = [] # finding similarity score by dot-prod between query(s) embed and each series-descrip embed
        for embeddings in embeddings_list:
            coef = [Query.inner_product(embeddings, series_dict[series_id]['embeddings']) for series_id in series_dict]
            similarity_column.append(coef)
        return similarity_column
    
    @staticmethod
    def preprocessing_softmax(embeddings_list):
        embeddings = np.array(embeddings_list)
        similarity = np.array(Query.get_col2(embeddings))
        logger.info(f"similarity score of the query:embeddings w.r.t series-description:embeddings => dim={similarity.shape} vals=\n{similarity}")
        X_train_1_tensor = torch.tensor(embeddings, dtype=torch.float32)
        X_train_2_tensor = torch.nn.functional.softmax(10 * torch.tensor(similarity, dtype=torch.float32), dim=-1)
        logger.info(f"creating DataLoader with train data with embeddings[dim:{X_train_1_tensor.shape}] & softmax(similarity_scores)[dim:{X_train_2_tensor.shape}]")
        dataset = TensorDataset(X_train_1_tensor, X_train_2_tensor)
        return DataLoader(dataset, batch_size=128, shuffle=True)
    
    @staticmethod
    def get_embeddings(text):
        response = embedding(text)
        return response.data[0].embedding

    def predict_wg(self):
        text_embeddings = Query.get_embeddings_list([self.enhanced_query])
        embeddings = text_embeddings[self.enhanced_query]
        logger.info("(i) Enhanced Query converted to Embeddings!")
        test_dataloader = Query.preprocessing_softmax([embeddings])
        logger.info("(ii) Applied Softmax-Preprocessing onto Embeddings!")
        label_list = []
        with torch.no_grad():
            for X1, X2 in test_dataloader:
                X1, X2 = X1.to(self.device), X2.to(self.device)
                outputs = self.model(X1, X2)
                logger.debug(f"NN-Router model: [inp:(queryEmbed.dim:{X1.shape},softSimScore.dim{X2.shape})] => [out(dim:{outputs.shape})=\n{outputs}]")
                _, top_indices = outputs.topk(5, dim=1) # a fn inside PyTorch.Tensors to get top k vals in resp dim
                predicted_labels = self.original_labels_mapping[top_indices.cpu().numpy()]
                label_list = predicted_labels
        logger.info(f"(iii) Found out all prediction_labels: {label_list} based on NN-Router model with existing weights!")
        self.wg = label_list[0]
        
    def get_question_context_faiss(self, batch, k, use_context=False):
        logger.info(f"=> called with batch.size={len(batch)} (no.of series + summaries inclusive), k={k}, use_context={use_context}")
        try:
            faiss_index, faiss_index_to_data_mapping, source_mapping, embedding_mapping = get_faiss_batch_index(batch)
            logger.info(f"---> get_faiss_batch_index() completed for given batch! Sample Entriy for Index[0]: source_mapping = {source_mapping[0]}, embedding_mapping.size = {embedding_mapping[0].shape}, \nfaiss_index_to_data_mapping = {repr(faiss_index_to_data_mapping[0])}")
            result = find_nearest_neighbors_faiss(self.query, faiss_index, faiss_index_to_data_mapping, k, source_mapping=source_mapping, embedding_mapping=embedding_mapping, context=self.context if use_context else None)
            logger.info(f"---> find_nearest_neighbors_faiss() completed with above params! \n {k}-Closest Results:")
            for entry in result:
                logger.debug(f"\t\tindex:{entry[0]}, source:{entry[2]}, embedding.size:{entry[3].shape}, text:\n{repr(entry[1])}")
            if isinstance(result, list):
                self.context = [f"\nRetrieval {i+1}:\n...{data}...\nThis retrieval is performed from the document 3GPP {source}.\n" for i, (_, data, source, _) in enumerate(result)]
                self.context_source = [f"Index: {index}, Source: {source}" for index, _, source, _ in result]
            else:
                self.context = result
        except Exception as e:
            print(f"An error occurred while getting question context: {e}")
            print(traceback.format_exc())
            self.context = "Error in processing"
    
    def validate_context(self, model_name='gpt-4o-mini', k=10, UI_flag=True):
        self.context = validator_RAG(self.question, self.context, model_name=model_name, k=k, UI_flag=UI_flag)
        
    def get_3GPP_context(self, k=10, model_name='gpt-4o-mini', validate_flag=True, UI_flag=False):
        self.predict_wg()
        logger.info(f"1. initital predict_wg() completed! predicted series no.s = {sorted(self.wg)}")
        
        doc_ds = get_documents(self.wg)
        logger.info(f"2. get_documents() completed! Total no.of docs within series{self.wg} = {len(doc_ds)}")
        
        Document_ds = [chunk_doc(doc) for doc in doc_ds]
        total_no_chunks = sum(len(doc_list) for doc_list in Document_ds)
        logger.info(f"3. chunking() completed! Database size after chucking each doc = {total_no_chunks}, sample chunk_entry: \n{Document_ds[0][0]}")
        
        series_doc = {'Summaries': []}
        for series_number in self.wg:
            # first 2 char of the file name, represents the series no. to which it belongs
            series_doc[f'Series{series_number}'] = [doc for doc in Document_ds if doc[0]['source'][:2].isnumeric() and int(doc[0]['source'][:2]) == series_number]
        series_doc['Summaries'] = [doc for doc in Document_ds if not doc[0]['source'][:2].isnumeric()]
        logger.info("4. categoriezed all doc chucks w.r.t to their series numbers!")

        series_docs = get_embeddings(series_doc)
        embedded_docs = [serie for serie in series_docs.values()]
        self.get_question_context_faiss(batch=embedded_docs, k=10, use_context=False)
        logger.info("5. get_question_context_faiss(use_context=False) completed! question_context is now rephrased with 'Retrieval [no]: [doc-chunk-text] from 3GPP document [source-file-name]' for all above entries.")
        self.candidate_answers(model_name=model_name, UI_flag=UI_flag)
        logger.info(f"6. candidate_answers() completed! question context enhanced with possible answers. enhanced_query:\n{repr(self.enhanced_query)}")

        old_list = self.wg
        self.predict_wg()
        logger.info(f"7. checking predict_wg() completed! newly predicted series no.s = {sorted(self.wg)} vs old predictions = {sorted(old_list)}")

        new_list = [series_number for series_number in self.wg if series_number not in old_list] # set difference
        new_doc_ds = get_documents(new_list) # fetch documents of newly predicted series
        new_Document_ds = [chunk_doc(doc) for doc in new_doc_ds]
        new_series = {}
        for series_number in new_list:
            new_series[f'Series{series_number}'] = [doc for doc in new_Document_ds if doc[0]['source'][:2].isnumeric() and int(doc[0]['source'][:2]) == series_number]
        
        logger.info(f"new series no.s that were not present in previously predicted series no.s = {new_series.keys()}")
        new_series = get_embeddings(new_series) # getting doc-chucks of [new-series-no] that are not present [in old-series-no]
        # copy the common series's data{'text','source','embeddings'} that are in common with newly predicted series
        old_series = {'Summaries': series_docs['Summaries'], **{f'Series{series_number}': series_docs[f'Series{series_number}'] for series_number in self.wg if series_number in old_list}}
        
        embedded_docs = [serie for serie in new_series.values()] + [serie for serie in old_series.values()] # combine all doc-chunks of newly pred series-no.s
        if validate_flag:
            logger.info(f"validate_flag is set to True, thus performing 'get_question_context_faiss()' with k = 2 times prev val:{k} = {2*k}")
            self.get_question_context_faiss(batch=embedded_docs, k=2*k, use_context=True)
        else:  
            self.get_question_context_faiss(batch=embedded_docs, k=k, use_context=True)
        logger.info("8. get_question_context_faiss(use_context=True) based on new predictions completed! question_context is now rephrased with 'Retrieval [no]: [doc-chunk-text] from 3GPP document [source-file-name]' for all above entries.")
        
        if validate_flag:
            self.validate_context(model_name=model_name, k=k, UI_flag=UI_flag)
            logger.info(f"spl.Task: validate_context() completed! question_context after validation:\n{repr(self.context)}")


    async def get_online_context(self, model_name='gpt-4o-mini', validator_flag= True, options=None):
        if options is None:
            querytoOSINT = f"""Rephrase the following question so that it can be a concise google search query to find the answer to my original question (O.S.I.N.T. syle)

        {self.question}"""
        else:
            querytoOSINT = f"""Rephrase the following multiple choice question so that it can be a concise google search query to find the answer to my original question (O.S.I.N.T. syle)

            {self.question}
    Answer options:
    {options}"""
        osintquery = await a_submit_prompt_flex(querytoOSINT, model=model_name)
        print("_"*100)
        print(osintquery)
        try:
            online_info = await fetch_snippets_and_search(query= osintquery.rstrip('"'), question=self.question, model_name=model_name, validator=validator_flag, UI=False)     
        except:
            online_info = await fetch_snippets_and_search(query= self.question, question=self.question, model_name=model_name, validator=validator_flag, UI=False)

        return online_info
    
    async def get_online_context_UI(self, model_name='gpt-4o-mini', validator_flag= True, options=None):
        if options is None:
            querytoOSINT = f"""Rephrase the fallowing question so that it can be a concise google search query to find the answer to my original question (O.S.I.N.T. syle)

        {self.question}"""
        else:
            querytoOSINT = f"""Rephrase the fallowing multiple choice question so that it can be a concise google search query to find the answer to my original question (O.S.I.N.T. syle)

            {self.question}
    Answer options:
    {options}"""
        osintquery = await a_submit_prompt_flex_UI(querytoOSINT, model=model_name)
        print("_FA"*100)
        print(osintquery)
        try:
            online_info = await fetch_snippets_and_search(query= osintquery.rstrip('"'), question=self.question, model_name=model_name, validator=validator_flag, UI=True)     
        except:
            online_info = await fetch_snippets_and_search(query= self.question, question=self.question, model_name=model_name, validator=validator_flag, UI=True)


        return online_info
