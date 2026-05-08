from pathlib import Path
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
import structlog


log = structlog.get_logger(__name__)

def load_documents(data_folder: str="data") -> list[Document]:
    
    documents: list[Document] = []

    try:
        p = Path(data_folder)
        files = [f for f in p.iterdir() if f.is_file()]
        log.info("loader.start", file_count=len(files))
        try:

            for item in files:
                if item.suffix.lower() == ".pdf":
                    document = PyPDFLoader(item)
                    documents += document.load()
                else:
                    log.debug("loader.skip", file=item.name)
                    continue
        except Exception as e:
            log.error("loader.error", file=item.name, error=str(e))

    except FileNotFoundError as e:
        print(f"The folder '{data_folder}' does not exist.")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)
    log.info("loader.done", total=len(documents))
    return documents

    
    
load_documents()