import abc
import streamlit as st
import pandas as pd
import csv


class Singleton(abc.ABCMeta):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

def show_user_file(uploaded_file):
    file_container = st.expander("Your CSV file :")
    read_file = csv.reader("./assets/" + uploaded_file.name, delimiter=',')
    length = 0
    for row in read_file:
        if len(row) > length:
            length = len(row)
    header = ['item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7']
    for i in range(length - 5):
        header.append('x' + str(i + 1))
        header.append('y' + str(i + 1))
        header.append('z' + str(i + 1))
        header.append('w' + str(i + 1))
    df = pd.read_csv("./assets/" + uploaded_file.name, names=header)
    df.to_csv('./assets/output.csv', index=False)

    shows = pd.read_csv('./assets/output.csv')
    uploaded_file.seek(0)
    file_container.write(shows)

def handle_upload():
    uploaded_file = st.file_uploader("upload", type="csv", label_visibility="collapsed")  # noqa: E501
    if uploaded_file is not None:
        show_user_file(uploaded_file)
        return pd.read_csv('./assets/output.csv')
    else:
        st.session_state["reset_chat"] = True

def disable_mainmenu():
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

def disable_sidebar():
    hide_sidebar_style = """
        <style>
        #sidebar {visibility: hidden;}
        </style>
        """
    st.markdown(hide_sidebar_style, unsafe_allow_html=True)
