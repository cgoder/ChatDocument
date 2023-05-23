import streamlit as st


def view_document(client):
    response = client.table("documents").select("metadata").execute()
    # 使用列表推导式和集合查找不重复的文件名
    file_names = list({d['metadata']['file_name'] for d in response.data})
    # st.write(file_names)
    for file_name in file_names:
        if st.button(file_name[:50].replace("\n", " ")):
            continue