import chromadb
from chromadb.utils import embedding_functions
from LLM import generate_response
from prompt import prompt1

class RAG:
    def __init__(self):
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-mpnet-base-v2"
                )
        
        self.client = chromadb.PersistentClient(path="./vector_db/")
        self.collection = self.client.get_or_create_collection(
            name="tours_collection",
            embedding_function=self.embedding_function
        )

        self.n_results = 5

        self.tools = None


    def query_tour(self, query, n_results):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

        
    def build_tour_prompt(self, user_query, n_results=5):
        """
        Query ChromaDB and create RAG prompt for tour recommendation.
        """

        results = self.query_tour(
            query=user_query,
            n_results=n_results
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        context_list = []

        for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
            context = f"""
            รายการที่ {i}
            ชื่อโปรแกรมทัวร์: {meta.get("program_tour", "ไม่พบข้อมูล")}
            ภูมิภาค: {meta.get("region", "ไม่พบข้อมูล")}
            ราคา: {meta.get("price", "ไม่พบข้อมูล")}
            URL: {meta.get("url", "ไม่พบข้อมูล")}
            รายละเอียด: {doc}
            """
            context_list.append(context)

            context_text = "\n".join(context_list)

            prompt = f"""
คุณคือระบบแนะนำโปรแกรมทัวร์ท่องเที่ยว (Tour Recommendation Assistant)

หน้าที่ของคุณคือแนะนำโปรแกรมทัวร์ที่เหมาะสมที่สุดจากข้อมูลที่ได้รับใน Context เท่านั้น

กฎการตอบ:
- ตอบโดยอ้างอิงเฉพาะข้อมูลใน Context เท่านั้น
- ห้ามสร้างข้อมูลขึ้นเอง
- หากไม่มีข้อมูลที่เกี่ยวข้องใน Context ให้ตอบว่า "ไม่พบข้อมูลทัวร์ที่ตรงกับความต้องการ"
- ตอบเป็นภาษาไทย
- ตอบสั้น กระชับ และอ่านง่าย
- แนะนำไม่เกิน {n_results} โปรแกรม
- หากมีราคา ให้แสดงราคา
- หากมี URL ให้แสดง URL

    Context:
    {context_text}

    คำถามผู้ใช้:
    {user_query}

    รูปแบบการตอบ:
    1. ชื่อโปรแกรมทัวร์:
    รายละเอียด:
    ราคา:
    URL:
    """
        return prompt


    def generate(self, query):
        prompt = self.build_tour_prompt(user_query=query, n_results=self.n_results)
        response = generate_response(prompt)
        return response







