from rag import RAG


rag = RAG()
query = "เที่ยวยเอเซียตะวันออกเฉียงใต้ใกล้ประเทศไทย"

result = rag.generate(query=query)

print(result)