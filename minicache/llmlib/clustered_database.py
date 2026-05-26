from __future__ import annotations

import logging
import pickle
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
from pydantic import BaseModel
import numpy as np
import math
from rank_bm25 import BM25Okapi
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ClusterDatabase(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    sentence_model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
    """ Sentence Transformer model for encoding the sentences """

    db_path: str = ""
    """Path to the database file."""

    clusters: Dict[int, List[Tuple[Any]]] = defaultdict(dict)
    """ Stores the data as clusters """

    cluster_centers: Dict[int, Tuple[Any]] = defaultdict(dict)
    """ Stores the cluster centers """

    max_cluster_id: int = -1

    # Similarity values for LASER
    # 0.9 for LASER expts
    # 0.8 for standalone prompt expts
    prompt_similarity_value: int = 0.8
    """ Prompt similarity value for clustering """

    # 0.9 for LASER expts
    # 0.75 for standalone prompt expts
    output_similarity_value: int = 0.75
    """ Output similarity value for clustering """


    def __init__(self, **kwargs: Any):
        """ Override init to support instantiation by position for backward compat. """
        super().__init__(**kwargs)
        self.load_db()

    
    def dump_db(self):
        """Dump the database to a file."""
        with open(self.db_path, "wb") as f:
            dump_dict = {"clusters": self.clusters, 
                         "cluster_centers": self.cluster_centers, 
                         "max_cluster_id": self.max_cluster_id}
            pickle.dump(dump_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


    def load_db(self):
        """Load the database from the file."""
        try:
            with open(self.db_path, "rb") as f:
                dump_dict = pickle.load(f)
                self.clusters = dump_dict["clusters"]
                self.cluster_centers = dump_dict["cluster_centers"]
                self.max_cluster_id = dump_dict["max_cluster_id"]
        except:
            logger.error(f"Error decoding Database file: {self.db_path}")


    def get_cluster_id(self, index: int):
        if index == -1:
            return -1
        cluster_id = np.array(list(self.clusters.keys()))
        return cluster_id[index]
    
    
    def get_output_embedding(self, output):
        if output is None:
            return None
        
        elif isinstance(output, str):
            return self.sentence_model.encode(output).reshape(1,-1)
        
        elif isinstance(output, dict):
            output_embedding = []
            for _, value in output.items():
                output_embedding.append(self.sentence_model.encode(value))
            return np.array(output_embedding)
        

    def get_output_similarity(self, center, output):
        if len(center.shape) == 1:
            center = center.reshape(1,-1)

        if output.shape[0] == 1:
            return cosine_similarity(center, output)[0][0]
        else:
            if center.shape[0] != output.shape[0]:
                return 0.
            else:
                output_sim_array = np.array([cosine_similarity(center[i].reshape(1,-1), output[i].reshape(1,-1))[0][0] for i in range(center.shape[0])])
                num_sim = len(np.array(output_sim_array) > self.output_similarity_value)
                if num_sim == output.shape[0]:
                    return np.mean(output_sim_array)
                else:
                    return 0.


    def find_similar_cluster(self, prompt_embedding, output_embedding=None, cluster_ids=[]):
        """ Computes cosine similarity to all the clusters and returns the cluster with the highest similarity """
        prompt_similarity_array = np.array([])
        output_similarity_array = np.array([])
        prompt_similarity_clusters = []
        output_similarity_clusters = []

        for cluster_id in self.clusters.keys():
            if (len(self.clusters.keys()) == len(cluster_ids)) or (str(cluster_id) not in cluster_ids):
                prompt_similarity_clusters.append(cluster_id)
                cluster_center = self.cluster_centers[cluster_id]

                prompt_similarity = cosine_similarity(cluster_center[0].reshape(1,-1), prompt_embedding.reshape(1,-1))[0][0]
                prompt_similarity_array = np.append(prompt_similarity_array, prompt_similarity)

                if output_embedding is not None:
                    output_similarity_clusters.append(cluster_id)
                    output_similarity = self.get_output_similarity(cluster_center[1], output_embedding)
                    output_similarity_array = np.append(output_similarity_array, output_similarity)     
        
        # Find the clusters whose similarity is more than the threshold
        print(f"Prompt Similarity Array: {prompt_similarity_array} | Output Similarity Array: {output_similarity_array}")
        prompt_similar_cluster_indices_tmp = np.where(np.array(prompt_similarity_array) > self.prompt_similarity_value)[0]
        prompt_similar_cluster_indices = [prompt_similarity_clusters[i] for i in prompt_similar_cluster_indices_tmp]

        output_similar_cluster_indices_tmp = np.where(np.array(output_similarity_array) > self.output_similarity_value)[0]
        output_similar_cluster_indices = [output_similarity_clusters[i] for i in output_similar_cluster_indices_tmp]

        print(f"Prompt Similar Cluster Indices: {prompt_similar_cluster_indices} | Output Similar Cluster Indices: {output_similar_cluster_indices}")
        return prompt_similar_cluster_indices, prompt_similarity_array, output_similar_cluster_indices, output_similarity_array
        

    def lexical_score(self, prompt, corpus):
        tokenized_corpus = [doc.split(" ") for doc in corpus]

        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = prompt.split(" ")
        return np.mean(bm25.get_scores(tokenized_query))
    

    def corpus_difference(self, prompt, corpus):
        similarity = []
        for doc in corpus:
            similarity.append(SequenceMatcher(None, prompt, doc).ratio())
        return np.mean(similarity)


    def set_db(self, key: Any, value: Any, cluster_ids: List = []):
        """ Set the database value for a corresponding key """
        print("\n ### Set DB ###\n")

        prompt_embedding = self.sentence_model.encode(value['prompt']).reshape(1,-1)
        # output_embedding = self.sentence_model.encode(value['output'])
        output_embedding = self.get_output_embedding(value['output'])

        if len(self.clusters) == 0:
            self.max_cluster_id = 0
            self.clusters[self.max_cluster_id] = [(prompt_embedding, key, value, output_embedding)]
            self.cluster_centers[self.max_cluster_id] = (prompt_embedding, output_embedding)
            self.dump_db()
            return 1, self.max_cluster_id
        
        prompt_similar_cluster_indices, prompt_similar_cluster_value, output_similar_cluster_indices, output_similar_cluster_value = self.find_similar_cluster(prompt_embedding, output_embedding, cluster_ids)
        
        # Choose the similar cluster with the highest similarity
        common_similar_cluster = np.intersect1d(prompt_similar_cluster_indices, output_similar_cluster_indices)
        if len(common_similar_cluster) > 0:
            # Choose the cluster with the highest sum similarity
            max_sum = -float('inf')
            max_cl_id = None
            for cl_id in common_similar_cluster:
                index_prompt = prompt_similar_cluster_indices.index(cl_id)
                index_output = output_similar_cluster_indices.index(cl_id)

                current_sum = prompt_similar_cluster_value[index_prompt] + output_similar_cluster_value[index_output]
                if current_sum > max_sum:
                    max_sum = current_sum
                    max_cl_id = cl_id

            max_similarity_cluster = max_cl_id

            print(f"Max Similarity Cluster: {max_similarity_cluster}")

            # Check if exact match is already present in the cluster wrt prompt as well as output
            # If present, do not add it again to the database
            # TODO: Is there any easier way to check if exact prompt was used before?
            for elem in self.clusters[max_similarity_cluster]:
                prompt_similarity = cosine_similarity(elem[0].reshape(1,-1), prompt_embedding.reshape(1,-1))[0][0]
                out_similarity = self.get_output_similarity(elem[3], output_embedding) 
                
                if (math.fabs(prompt_similarity - 1.0) < 1e-5) and (math.fabs(out_similarity - 1.0) < 1e-5):
                    print("Exact match found, not adding to the database")
                    return 0, max_similarity_cluster
            
            # Append the new prompt to the cluster
            self.clusters[max_similarity_cluster].append((prompt_embedding, key, value, output_embedding))
            new_prompt_cluster_center = np.mean([t[0] for t in self.clusters[max_similarity_cluster]], axis=0)
            new_output_cluster_center = np.mean([t[3] for t in self.clusters[max_similarity_cluster]], axis=0)
            self.cluster_centers[max_similarity_cluster] = (new_prompt_cluster_center, new_output_cluster_center)

            print("Appended to cluster")

        else:
            # If no cluster has similarity greater than 90%, create a new cluster and assign the prompt template to it
            print("Creating new cluster")
            self.max_cluster_id += 1
            self.clusters[self.max_cluster_id] = [(prompt_embedding, key, value, output_embedding)]
            self.cluster_centers[self.max_cluster_id] = (prompt_embedding, output_embedding)
            max_similarity_cluster = self.max_cluster_id

        self.dump_db()
        return 1, max_similarity_cluster

    
    def get_db_with_keys(self, key: Any):
        """ Get the database value for the corresponding key """
        print("\n ### Get DB ###\n")
        max_similar_cluster = key
        print(f"Max Similar Cluster: {max_similar_cluster}")
        if max_similar_cluster != -1:
            inp_out_list = {elem[1]:elem[2] for elem in self.clusters[max_similar_cluster]}
            print(f"Number of records in the cluster: {len(inp_out_list)}")
            return max_similar_cluster, inp_out_list
        else:
            return None, None


    def get_db(self, value: Any):
        """ Get the database value for the corresponding key """
        print("\n ### Get DB ###\n")
        max_similar_cluster_idx = -1

        if type(value) == str:
            prompt_embedding = self.sentence_model.encode(value)
            output_embedding = None
        else:
            prompt_embedding = self.sentence_model.encode(value['prompt'])
            output_embedding = self.get_output_embedding(value["output"])
            # output_embedding = self.sentence_model.encode(value['output'])

        prompt_similar_cluster_indices, prompt_similar_cluster_value, output_similar_cluster_indices, output_similar_cluster_value = self.find_similar_cluster(prompt_embedding, output_embedding)

        # Find the most similar cluster
        if len(prompt_similar_cluster_value) > 0:
            if output_embedding is None:

                try:
                    max_similar_cluster_idx = max(prompt_similar_cluster_indices, key=lambda idx: prompt_similar_cluster_value[idx])
                except:
                    pass
            else:
                common_similar_cluster = np.intersect1d(prompt_similar_cluster_indices, output_similar_cluster_indices)
                if len(common_similar_cluster) > 0:
                    max_similar_cluster_idx = common_similar_cluster[np.argmax(prompt_similar_cluster_value[common_similar_cluster] + output_similar_cluster_value[common_similar_cluster])]


        # return the input-output list of the most similar cluster
        max_similar_cluster = self.get_cluster_id(max_similar_cluster_idx)
        print(f"Max Similar Cluster: {max_similar_cluster}")
        if max_similar_cluster != -1:
            inp_out_list = {elem[1]:elem[2] for elem in self.clusters[max_similar_cluster]}
            print(f"Number of records in the cluster: {len(inp_out_list)}")
            return max_similar_cluster, inp_out_list
        else:
            return None, None


    def redistribute_db(self, cluster_to_redistribute: int, prompts_to_move: List[Tuple, Any]):
        indices_to_remove = []
        elem_to_remove = []
        print("Cluster to redistribute: ", cluster_to_redistribute)
        for i,elem in enumerate(self.clusters[cluster_to_redistribute]):
            for p in prompts_to_move:
                if elem[1] == p[0]:
                    indices_to_remove.append(i)
                    elem_to_remove.append(elem)
        
        # Remove few elements and adjust the cluster center
        print("Removing indices: ", indices_to_remove)
        self.clusters[cluster_to_redistribute] = [elem for i,elem in enumerate(self.clusters[cluster_to_redistribute]) if i not in indices_to_remove]
        new_prompt_cluster_center = np.mean([t[0] for t in self.clusters[cluster_to_redistribute]], axis=0)
        new_output_cluster_center = np.mean([t[3] for t in self.clusters[cluster_to_redistribute]], axis=0)
        self.cluster_centers[cluster_to_redistribute] = (new_prompt_cluster_center, new_output_cluster_center)

        # Add the elements removed to a new cluster
        print("Creating a new cluster for the removed elements with cluster ID: ", self.max_cluster_id+1)
        self.max_cluster_id += 1
        self.clusters[self.max_cluster_id] = []
        self.clusters[self.max_cluster_id].extend(elem_to_remove)
        new_prompt_cluster_center = np.mean([t[0] for t in self.clusters[self.max_cluster_id]], axis=0)
        new_output_cluster_center = np.mean([t[3] for t in self.clusters[self.max_cluster_id]], axis=0)
        self.cluster_centers[self.max_cluster_id] = (new_prompt_cluster_center, new_output_cluster_center)

        self.dump_db()


    def len(self, key: Any):
        """ Checks the number of records in the database with the same key """
        print("\n ### len DB ###\n")
        return len(self.clusters[key])


    '''
    def len(self, value: Any):
        """ Checks the number of records in the database with the same key """
        print("\n ### len DB ###\n")
        max_similar_cluster_idx = -1

        if type(value) == str:
            prompt_embedding = self.sentence_model.encode(value)
            output_embedding = None
        else:
            prompt_embedding = self.sentence_model.encode(value['prompt'])
            output_embedding = self.get_output_embedding(value["output"])
            # output_embedding = self.sentence_model.encode(value['output'])

        prompt_similar_cluster_indices, prompt_similar_cluster_value, output_similar_cluster_indices, output_similar_cluster_value = self.find_similar_cluster(prompt_embedding, output_embedding)

        # Find the most similar cluster
        try:
            if len(prompt_similar_cluster_value) > 0:
                if output_embedding is None:
                    # if len(prompt_similar_cluster_indices) > 1:
                    #     for idx in prompt_similar_cluster_indices:
                    #         print(f"Cluster ID: {idx} | Lexical Score: {self.lexical_score(value['prompt'], idx)}")
                    #     max_similar_cluster = max(prompt_similar_cluster_indices, key=lambda idx: self.lexical_score(value['prompt'], idx))
                    # else:
                    #     max_similar_cluster = max(prompt_similar_cluster_indices, key=lambda idx: prompt_similar_cluster_value[idx])
                    max_similar_cluster_idx = max(prompt_similar_cluster_indices, key=lambda idx: prompt_similar_cluster_value[idx])
                else:
                    common_similar_cluster = np.intersect1d(prompt_similar_cluster_indices, output_similar_cluster_indices)
                    print(common_similar_cluster)
                    if len(common_similar_cluster) > 0:
                        max_similar_cluster_idx = common_similar_cluster[np.argmax(prompt_similar_cluster_value[common_similar_cluster] + output_similar_cluster_value[common_similar_cluster])]
        except:
            pass
        
        max_similar_cluster = self.get_cluster_id(max_similar_cluster_idx)
        print(f"Max Similar Cluster: {max_similar_cluster}")
        if max_similar_cluster != -1:
            return len(self.clusters[max_similar_cluster])
        else:
            return 0
    '''