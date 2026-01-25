import chromadb

# For a local database
client = chromadb.PersistentClient(path="/Users/itay/GitHub/printer_ai_support_voice_bot/agent_graph/chroma_store")

# For a running Chroma server
# client = chromadb.HttpClient(host='localhost', port=8000)

# List all collections
collections = client.list_collections()

for collection in collections:
    # In newer versions of Chroma, this returns a list of strings
    # In older versions, it returns a list of collection objects
    print(collection)