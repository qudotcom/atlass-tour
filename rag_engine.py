import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

load_dotenv()

class AtlasBrain:
    def __init__(self):
        self.vector_db = None
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        self.embeddings = OpenAIEmbeddings()
        
    def load_data(self):
        """
        Simule le chargement des données de Personne B[cite: 35, 36].
        Normalement, on lirait un Excel/JSON ici.
        """
        raw_data = [
            {
                "Nom": "Médina de Fès",
                "Description": "La plus grande zone piétonne au monde, un labyrinthe historique.",
                "Accessibilite": "Non",
                "Type": "Foule/Historique",
                "Metadonnees": "Attention aux ruelles étroites et aux escaliers."
            },
            {
                "Nom": "Jardin Majorelle",
                "Description": "Un jardin botanique paisible avec une villa art déco bleue.",
                "Accessibilite": "Oui",
                "Type": "Calme/Nature",
                "Metadonnees": "Rampes disponibles pour fauteuils roulants."
            },
             {
                "Nom": "Place Jemaa el-Fna",
                "Description": "Place célèbre de Marrakech, animée avec des charmeurs de serpents.",
                "Accessibilite": "Oui",
                "Type": "Foule/Spectacle",
                "Metadonnees": "Surface plate mais très fréquentée."
            }
        ]
        
        # Transformation en documents LangChain pour le RAG
        documents = []
        for item in raw_data:
            content = f"Lieu: {item['Nom']}. Description: {item['Description']}. Type: {item['Type']}."
            # On ajoute les métadonnées pour le filtrage (Smart Planner) [cite: 53]
            meta = {"accessibilite": item['Accessibilite'], "nom": item['Nom']}
            documents.append(Document(page_content=content, metadata=meta))
            
        print(f"📚 {len(documents)} lieux chargés dans la mémoire.")
        return documents

    def initialize_brain(self):
        """Vectorise les données et crée la base de données Chroma [cite: 52]"""
        docs = self.load_data()
        # Création du VectorStore en mémoire
        self.vector_db = Chroma.from_documents(
            documents=docs, 
            embedding=self.embeddings,
            collection_name="atlas_places"
        )
        print("🧠 Cerveau initialisé et vectorisé.")

    def ask_atlas(self, question, filter_accessibility=False):
        """
        Pose une question au moteur RAG.
        Gère le filtre d'accessibilité (Semaine 3 logic anticipation) [cite: 53, 71]
        """
        if not self.vector_db:
            return "Erreur: Cerveau non initialisé."

        # Définition du Prompt "Guide Expert" [cite: 70]
        prompt_template = """Tu es un guide expert marocain pour l'application Atlas Tour.
        Utilise les éléments de contexte suivants pour répondre à la question.
        Si tu ne sais pas, dis-le simplement. Sois chaleureux mais précis.
        
        Contexte: {context}
        
        Question: {question}
        Réponse:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Configuration du Retriver
        search_kwargs = {"k": 3}
        
        # Filtrage strict si mobilité réduite demandée [cite: 53]
        if filter_accessibility:
            search_kwargs["filter"] = {"accessibilite": "Oui"}

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(search_kwargs=search_kwargs),
            chain_type_kwargs={"prompt": PROMPT}
        )

        return qa_chain.invoke(question)["result"]

# --- ZONE DE TEST (Pour valider le travail de la Semaine 2) ---
if __name__ == "__main__":
    bot = AtlasBrain()
    bot.initialize_brain()
    
    print("\n--- TEST 1: Question Générale ---")
    print(bot.ask_atlas("Qu'est-ce qu'on peut voir à Fès ?"))
    
    print("\n--- TEST 2: Filtre Accessibilité (Le Crash Test) [cite: 53] ---")
    # Si je demande des lieux accessibles, Fès (marqué Non) ne doit PAS apparaître.
    print(bot.ask_atlas("Quels lieux sont accessibles en fauteuil roulant ?", filter_accessibility=True))
