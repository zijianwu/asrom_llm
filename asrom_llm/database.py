from langchain.vectorstores import Milvus

from asrom_llm.utils import chunks

DATABASE_CHUNK_SIZE = 5000


class ModifiedMilvus(Milvus):
    @classmethod
    def from_documents(
        cls,
        documents,
        embedding,
        collection_name="medline_collection",
        **kwargs,
    ):
        """Return VectorStore initialized from documents and embeddings.

        Added ability to define a specific collection name.

        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(
            texts,
            embedding,
            collection_name=collection_name,
            metadatas=metadatas,
            **kwargs,
        )

    def add_documents(self, documents, **kwargs):
        """Run more documents through the embeddings and add to the vectorstore.
        Args:
            documents (List[Document]: Documents to add to the vectorstore.
        Returns:
            List[str]: List of IDs of the added texts.
        """
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        for text, metadata in chunks(DATABASE_CHUNK_SIZE, texts, metadatas):
            self.add_texts(text, metadata)
        return self

    @classmethod
    def from_texts(
        cls,
        texts,
        embedding,
        collection_name="medline_collection",
        metadatas=None,
        **kwargs,
    ):
        """Create a Milvus collection, indexes it with HNSW, and insert data.

        Added ability to define a specific collection name.

        Args:
            texts (List[str]): Text to insert.
            embedding (Embeddings): Embedding function to use.
            metadatas (Optional[List[dict]], optional): Dict metatadata.
                Defaults to None.
        Returns:
            VectorStore: The Milvus vector store.
        """
        try:
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
            )
            from pymilvus.orm.types import infer_dtype_bydata
        except ImportError:
            raise ValueError(
                "Could not import pymilvus python package. "
                "Please install it with `pip install pymilvus`."
            )
        # Connect to Milvus instance
        if not connections.has_connection("default"):
            connections.connect(**kwargs.get("connection_args", {"port": 19530}))
        # Determine embedding dim
        embeddings = embedding.embed_query(texts[0])
        dim = len(embeddings)
        # Generate unique names
        primary_field = "primary_field"  # "c" + str(uuid.uuid4().hex)
        vector_field = "vector_field"  # "c" + str(uuid.uuid4().hex)
        text_field = "text_field"  # "c" + str(uuid.uuid4().hex)
        fields = []
        # Determine metadata schema
        if metadatas:
            # Check if all metadata keys line up
            key = metadatas[0].keys()
            for x in metadatas:
                if key != x.keys():
                    raise ValueError(
                        "Mismatched metadata. "
                        "Make sure all metadata has the same keys and datatype."
                    )
            # Create FieldSchema for each entry in singular metadata.
            for key, value in metadatas[0].items():
                # Infer the corresponding datatype of the metadata
                dtype = infer_dtype_bydata(value)
                if dtype == DataType.UNKNOWN:
                    raise ValueError(f"Unrecognized datatype for {key}.")
                elif dtype == DataType.VARCHAR:
                    # Find out max length text based metadata
                    # TODO: Choose better max_length based on the length of the text
                    max_length = 1_000  # 0
                    # for subvalues in metadatas:
                    #     max_length = max(max_length, len(subvalues[key]))
                    fields.append(FieldSchema(key, DataType.VARCHAR, max_length=max_length + 1))
                else:
                    fields.append(FieldSchema(key, dtype))

        # Find out max length of texts
        # TODO: Choose better max_length based on the length of the text
        max_length = 1_000  # 0
        # for y in texts:
        #     max_length = max(max_length, len(y))
        # Create the text field
        fields.append(FieldSchema(text_field, DataType.VARCHAR, max_length=max_length + 1))
        # Create the primary key field
        fields.append(FieldSchema(primary_field, DataType.INT64, is_primary=True, auto_id=True))
        # Create the vector field
        fields.append(FieldSchema(vector_field, DataType.FLOAT_VECTOR, dim=dim))
        # Create the schema for the collection
        schema = CollectionSchema(fields)
        # Create the collection
        collection = Collection(collection_name, schema)
        # Index parameters for the collection
        index = {
            "index_type": "IVF_SQ8",
            "metric_type": "L2",
            "params": {"nlist": 100},
        }
        # Create the index
        collection.create_index(vector_field, index)
        # Create the VectorStore
        milvus = cls(
            embedding,
            kwargs.get("connection_args", {"port": 19530}),
            collection_name,
            text_field,
        )
        # Add the texts.
        for text, metadata in chunks(DATABASE_CHUNK_SIZE, texts, metadatas):
            milvus.add_texts(text, metadata)

        return milvus
