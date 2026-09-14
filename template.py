import os
import pathlib

def create_reranking_structure():
    """Create the re-ranking project structure with empty files"""
    
    # Define the complete folder structure
    structure = {
        "": [  # Root directory files
            "app.py",
            "config.yaml",
            "requirements.txt",
            "test-nb.ipynb",
            "readme.md",
            ".env"
        ],
        "data/raw/insurance_docs": [
            "policy_terms.txt",
            "claim_procedure.txt"
        ],
        "data/embeddings": [
            "faiss_index/"
        ],
        "utils": [
            "loader.py",
            "retriever_faiss.py",
            "reranker_cross_encoder.py",
            "reranker_llm.py",
            "two_stage_retriever.py"
        ]
    }
    
    print("Creating Re-ranking Project structure...")
    
    # Create all folders and files
    for folder, items in structure.items():
        # Create folder if it doesn't exist
        if folder:
            os.makedirs(folder, exist_ok=True)
            print(f"📁 Created folder: {folder}/")
        
        # Create files within the folder
        for item in items:
            file_path = os.path.join(folder, item) if folder else item
            
            # Handle directories (ending with /)
            if item.endswith('/'):
                os.makedirs(file_path, exist_ok=True)
                print(f"📁 Created folder: {file_path}")
            else:
                # Create empty file
                pathlib.Path(file_path).touch()
                print(f"📄 Created file: {file_path}")
    
    print("\n✅ Re-ranking Project structure created successfully!")
    print("📂 All files and folders are now ready.")

if __name__ == "__main__":
    create_reranking_structure()