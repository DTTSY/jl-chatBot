from llama_index.readers.file import FlatReader
from pathlib import Path
from langchain_docling.loader import DoclingLoader
from langchain_community.document_loaders import PyPDFLoader
import asyncio

base_path = "G:/FTPFIle"
def parse_document(file_path: str):
    reader = FlatReader()
    # 拼接文件路径
    parsed_data = reader.load_data(Path().joinpath(base_path, file_path))
    return parsed_data

def parse_document_docling(file_path: str):
    path = f'{base_path}/{file_path}'
    loader = DoclingLoader(path)
    docs = loader.load()
    return docs

async def parse_document_PyPDF(file_path: str):
    path = f'{base_path}/{file_path}'
    loader = PyPDFLoader(path)
    docs = loader.load()
    parsed_data = docs[0].page_content
    return parsed_data


if __name__ == "__main__":
    # 测试代码
    file_paths = ['deepseekGRPO.pdf']  # 替换为实际的文件路径
    for file_path in file_paths:
        print(f'parse {file_path}')

        # parsed_data = parse_document(file_path)
        # print(parsed_data[0].text)  # 输出解析后的数据
        # parsed_data = parse_document_docling(file_path)
        parsed_data = asyncio.run(parse_document_PyPDF(file_path))
        print(parsed_data)  # 输出解析后的数据
