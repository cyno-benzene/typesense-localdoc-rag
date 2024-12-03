import os
from dotenv import load_dotenv
from pydantic.dataclasses import dataclass
from pydantic import BaseModel

load_dotenv()


PDF_DIRECTORY = "./downloaded_pdfs"
TYPESENSE_API_KEY = os.getenv('TYPESENSE_API_KEY')

def set_env():
    # print("------KEY:",str(os.getenv('GOOGLE_API_KEY')))
    # print("------KEY:",str(os.getenv('TYPESENSE_API_KEY')))

    os.environ['GOOGLE_API_KEY'] = str(os.getenv('GOOGLE_API_KEY'))
    os.environ['TYPESENSE_API_KEY'] = str(os.getenv('TYPESENSE_API_KEY'))

